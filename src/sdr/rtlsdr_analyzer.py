"""
# =============================================================================
#  OpenWX -- Open Weather Radiosonde Telemetry System
# =============================================================================
#
#  File   : rtlsdr_analyzer.py
#  Author : M.F. Guenther, DL2MF - DL2MF@darc.de
#  License: GNU General Public License v2.0 (GPL-2.0)
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; version 2 of the License.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# -----------------------------------------------------------------------------
#  Description
# -----------------------------------------------------------------------------
#
#  RTL-SDR spectrum analyzer and signal detector for OpenWX.
#
#  Provides SpectrumAnalyzer, which captures IQ samples via pyrtlsdr,
#  computes a Welch power spectral density estimate, and detects candidate
#  radiosonde signals as peaks above a configurable SNR threshold.
#
#  Detection pipeline:
#    pyrtlsdr (IQ capture) → scipy Welch PSD → peak detection
#      → bandwidth filter (2–30 kHz) → DetectedSignal list
#
#  The analyzer supports pause/resume to yield the USB device to rtl_fm
#  during active decoding. Continuous scanning runs in a background thread.
#
#  Decoder backend : rs1729 (RS41, DFM09, M10, iMet-C, ...)
#  Hardware        : RTL-SDR (RTL2832U-based receivers)
#
# =============================================================================
"""

import numpy as np
import logging
import time
import threading
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass
from scipy import signal as scipy_signal

try:
    from rtlsdr import RtlSdr
    RTLSDR_AVAILABLE = True
except ImportError:
    RTLSDR_AVAILABLE = False
    logging.warning("pyrtlsdr not available - RTL-SDR support disabled")


@dataclass
class DetectedSignal:
    """Represents a detected radiosonde signal"""
    frequency: float
    strength: float  # SNR in dB (peak power above noise floor)
    bandwidth: float
    timestamp: float
    # Absolute peak power at the signal bin (dBFS) and the scan's noise floor
    # (dBFS). strength == power_dbfs - noise_floor_dbfs. Kept separately so the
    # UI can show a real RSSI (power) distinct from SNR — without these, both
    # RSSI and SNR fell back to the same SNR value and displayed identically.
    power_dbfs: float = 0.0
    noise_floor_dbfs: float = 0.0
    

class SpectrumAnalyzer:
    """Analyzes spectrum and detects radiosonde signals"""

    # librtlsdr's own rtl_sdr.c reference tool reads in chunks of 16*16384
    # samples by default — a USB bulk-transfer size known to be reliable.
    # capture_spectrum() reads in chunks of this size (never "however many
    # samples make up N seconds", which can be megabytes at typical 2.4 MSPS
    # configs and risks LIBUSB_ERROR_NO_MEM).
    READ_CHUNK_SAMPLES = 16 * 16384  # 262144

    def __init__(self, config: dict, device_config: dict = None, frequency_blacklist: list = None):
        """
        Initialize spectrum analyzer
        
        Args:
            config: Full application config
            device_config: Specific device configuration (from rtlsdr.devices list)
                          If None, uses first device from config
            frequency_blacklist: List of frequencies (Hz) to ignore during detection
        """
        self.config = config
        self.logger = logging.getLogger('SpectrumAnalyzer')
        self.sdr = None
        self.running = False
        self.detected_signals: List[DetectedSignal] = []
        self.lock = threading.Lock()
        
        # Frequency blacklist (Hz) - ignore these frequencies in detection
        self.frequency_blacklist = frequency_blacklist or []
        
        # Get device configuration
        if device_config is None:
            # Backward compatibility: use first device or old format
            rtlsdr_config = config['sdr']['rtlsdr']
            if 'devices' in rtlsdr_config:
                device_config = rtlsdr_config['devices'][0]
            else:
                # Old config format
                device_config = rtlsdr_config
        
        self.device_config = device_config
        self.device_serial = device_config.get('serial', '0')
        
        # Configuration. Use .get with defaults — a device config missing
        # center_freq/sample_rate must not crash the scan loop (it used to raise
        # KeyError on every cycle and wedge the worker). The manager also fills
        # these defaults up front; this is defense-in-depth.
        self.center_freq = device_config.get('center_freq') or 404_000_000
        self.sample_rate = device_config.get('sample_rate') or 2_400_000
        self.fft_size = config['detection']['fft_size']
        self.detection_threshold = config['detection']['detection_threshold']
        self.scan_interval = config['receivers']['scan_interval']

        # Scan tuning: a much longer integration window before peak-picking
        # than a single FFT snapshot, real-world channel spacing for peak
        # min-distance/quantization, and a cap on how many candidates a
        # single scan pass can report.
        det_cfg = config['detection']
        # Default lowered 20 → 5 s: on a 4-device Pi the concurrent Welch
        # averaging inflated a "20 s" dwell to minutes of wall time (field:
        # first snapshots took 2-8 min after startup) and starved decoder
        # pipes. 5 s (~5 averaged batches) still cleans up the peak list well.
        self.scan_check_time = float(det_cfg.get('scan_check_time', 5.0))
        # Hard wall-clock cap on a single capture_spectrum() call. On a 4-dongle
        # Pi, CPU/USB contention inflates a nominal N-second dwell to minutes,
        # which (a) never completes the first scan and (b) leaves the worker
        # stuck in an unlocked libusb read so a concurrent close() SIGBUSes.
        # This cap guarantees capture returns promptly with whatever it has
        # averaged so far. Fixed low default, DECOUPLED from scan_check_time so
        # a high dwell (config or contention) can never stretch capture without
        # bound — if scan_check_time < cap the loop finishes normally first; if
        # it's higher, we just return a shorter (still valid) average.
        self.scan_max_wall_s = float(det_cfg.get('scan_max_wall_s', 8.0))
        self.channel_spacing_hz = float(det_cfg.get('channel_spacing_hz', 10_000))
        self.max_peaks = int(det_cfg.get('max_peaks', 10))

        # Reject candidate peaks that don't line up with the sonde band's
        # channel raster (e.g. 401.000-405.990 MHz in 10 kHz steps). A strong
        # nearby signal's skirt/sidelobe can produce a detectable local
        # maximum a few kHz off the true channel, which then gets identified
        # as a spurious "sonde" on a frequency that no real transmitter uses.
        # Real sondes drift up to ~3 kHz off their nominal channel; anything
        # drifting further is almost certainly not a genuine channel signal.
        self.check_frequency_raster = bool(det_cfg.get('check_frequency_raster', True))
        self.raster_tolerance_hz = float(det_cfg.get('raster_tolerance_hz', 3_000))

        # DC-spur guard: RTL-SDR tuners produce a persistent DC/1-f spur at or
        # near the tuned center frequency. Field-observed: RTL00004 showed a
        # permanent spike exactly at its 403.200 MHz center; RTL00003
        # repeatedly "detected" a phantom signal at 404.5813 MHz (18.7 kHz
        # below its 404.600 center), dft_detect found nothing, and the
        # bandwidth fallback started a 300 s phantom DFM decode every scan
        # cycle. auto_rx likewise excludes the DC region from peak-picking.
        # NOTE: this makes channels within ±dc_notch_hz of a device's
        # center_freq invisible TO THAT DEVICE — pick center frequencies
        # offset from active sonde channels (e.g. 404.605 instead of 404.600).
        self.dc_notch_hz = float(det_cfg.get('dc_notch_hz', 25_000))
        
    def initialize(self) -> bool:
        """Initialize RTL-SDR device"""
        if not RTLSDR_AVAILABLE:
            self.logger.error("RTL-SDR support not available")
            return False
            
        try:
            # Support both serial number and device index.
            # If serial is all digits treat as a raw device index;
            # otherwise resolve the serial to an index via librtlsdr.
            if self.device_serial.isdigit():
                device_index = int(self.device_serial)
                self.sdr = RtlSdr(device_index)
                self.logger.info(f"Opening RTL-SDR by index: {device_index}")
            else:
                device_index = RtlSdr.get_device_index_by_serial(self.device_serial)
                self.sdr = RtlSdr(device_index)
                self.logger.info(f"Opening RTL-SDR serial '{self.device_serial}' → index {device_index}")
            
            # Configure SDR
            self.sdr.sample_rate = self.sample_rate
            self.sdr.center_freq = self.center_freq
            
            gain = self.device_config.get('gain', 0)
            if gain == 0:
                self.sdr.gain = 'auto'
            else:
                self.sdr.gain = gain
                
            ppm = self.device_config.get('ppm_error', 0)
            if ppm != 0:
                self.sdr.freq_correction = ppm
            
            self.logger.info(f"RTL-SDR initialized: {self.center_freq/1e6:.3f} MHz, "
                           f"{self.sample_rate/1e6:.2f} MSPS, gain={self.sdr.gain}, serial={self.device_serial}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RTL-SDR: {e}")
            return False
    
    def close(self):
        """Close RTL-SDR device"""
        if self.sdr:
            self.sdr.close()
            self.sdr = None
            self.logger.info("RTL-SDR closed")
    
    def capture_spectrum(
        self,
        dwell_time: float = None,
        abort_check: Optional[Callable[[], bool]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Capture and analyze spectrum, integrated over `dwell_time` seconds
        (default: self.scan_check_time, e.g. 20s) rather than a single short
        FFT snapshot.

        CRITICAL: a 20s-dwell capture is now a long blocking call (previously
        a single snapshot took milliseconds). DeviceWorker's scan cycle only
        checks its manual-decode-pending flag before/after this call, so
        without a way to bail out mid-capture, a manual/priority decode
        request arriving during a scan could be stuck waiting up to the full
        dwell time — exactly the class of race this project has repeatedly
        had to fix elsewhere. `abort_check`, if given, is polled once per
        chunk; on a True result we stop early and return whatever was
        integrated so far (still a valid, if shorter, average) rather than
        block for the remainder of dwell_time.

        Averaging ~20s of spectrum before peak-picking gives a much cleaner
        peak list than a single-snapshot approach — weak/drifting candidates
        that don't clear the noise floor in one 2048-point FFT often do once
        variance is averaged down over many independent periodograms.

        Implementation note: rather than one giant read_samples() call sized
        for the full dwell time (e.g. 20s @ 2.4 MSPS would be ~48M complex
        samples — several hundred MB, too much for a Raspberry Pi, especially
        with multiple devices scanning at once), this reads a sequence of
        smaller chunks, computes a Welch PSD per chunk, and averages the
        *linear* power spectra (never average in dB — that's not
        mathematically the same and biases the result high) before the final
        dB conversion. This gives the same statistical benefit as one long
        capture while keeping peak memory bounded to a single chunk.

        Returns (frequencies, power_db)
        """
        if not self.sdr:
            raise RuntimeError("SDR not initialized")

        if dwell_time is None:
            dwell_time = self.scan_check_time

        # CRITICAL: chunk size must be a safe single USB bulk-transfer size,
        # NOT "however many samples make up 1 second at this sample rate".
        # That earlier approach requested 2.4M samples (4.8 MB) per
        # read_samples() call at a typical 2.4 MSPS device config, which
        # intermittently failed with LIBUSB_ERROR_NO_MEM ("Insufficient
        # memory") — especially with several RTL-SDR dongles sharing the
        # same USB controller's usbfs transfer-memory budget. 16*16384 =
        # 262144 samples is librtlsdr's own rtl_sdr.c reference tool's
        # default bulk-transfer chunk size, a value known to be safe.
        chunk_samples = self.READ_CHUNK_SAMPLES
        chunk_duration_s = chunk_samples / self.sample_rate
        num_chunks = max(1, round(dwell_time / chunk_duration_s))

        # CRITICAL: running welch() on every single ~0.109s USB-safe chunk
        # (num_chunks ≈ 183 for a 20s dwell at 2.4 MSPS) means ~183 * ~255
        # internal FFTs per device per cycle — ~47,000 FFTs, ×4 devices
        # scanning concurrently ≈ 188,000 FFTs every 20s. That sustained CPU
        # load was observed in the field to inflate the *actual* wall-clock
        # duration of a single "20s" capture to 80-100s+ under 4-device
        # contention (get_spectrum() stayed empty that whole time), and is a
        # strong suspect for intermittent decode failures elsewhere on the
        # same host: 3 sibling devices burning CPU on redundant FFTs can
        # starve the rtl_fm→decoder pipe of scheduling long enough to break
        # bit sync without the decoder process erroring out.
        # Fix: batch several USB-safe read chunks into one ~1s buffer before
        # each welch() call — same read_samples() transfer size (still USB-
        # safe), same total samples averaged, ~9x fewer welch()/FFT calls.
        chunks_per_batch = max(1, round(1.0 / chunk_duration_s))

        freqs = None
        psd_accum = None
        batches_done = 0
        chunks_done = 0
        aborted = False
        capture_start = time.time()
        while chunks_done < num_chunks and not aborted:
            # Hard wall-clock cap: under 4-device CPU/USB contention a nominal
            # dwell can stretch to minutes and hang the worker in a libusb read.
            # Bail with whatever we've averaged so far — a shorter average is a
            # valid spectrum, and returning promptly is what keeps the device
            # closeable (no SIGBUS) and the first scan actually completing.
            if time.time() - capture_start > self.scan_max_wall_s:
                self.logger.debug(
                    f"capture_spectrum: wall-clock cap ({self.scan_max_wall_s:.0f}s) hit "
                    f"after {chunks_done}/{num_chunks} chunks — returning partial average"
                )
                break
            batch = []
            for _ in range(min(chunks_per_batch, num_chunks - chunks_done)):
                # CRITICAL: check BEFORE calling read_samples(), not just
                # after. Checking only after each read left a window at the
                # start of every new batch — welch() on the previous batch
                # takes a moment, and a concurrent manual/priority decode's
                # _teardown_scan() (self.sdr.close(); self.sdr = None) can
                # land in exactly that window, so the next read_samples()
                # call dereferences a None self.sdr. Observed in the field:
                # "AttributeError: 'NoneType' object has no attribute
                # 'read_samples'", ~1s after the device had already closed.
                if self.sdr is None or (abort_check is not None and abort_check()):
                    aborted = True
                    break
                batch.append(self.sdr.read_samples(chunk_samples))
                chunks_done += 1

            if not batch:
                break

            batch_samples = np.concatenate(batch) if len(batch) > 1 else batch[0]

            batch_freqs, batch_psd = scipy_signal.welch(
                batch_samples,
                fs=self.sample_rate,
                nperseg=self.fft_size,
                scaling='density',
                return_onesided=False
            )

            if psd_accum is None:
                freqs = batch_freqs
                psd_accum = batch_psd
            else:
                psd_accum += batch_psd
            batches_done += 1

            if aborted:
                self.logger.debug(
                    f"capture_spectrum: abort_check triggered after {chunks_done}/{num_chunks} chunks "
                    f"({batches_done} welch batches)"
                )

        if psd_accum is None:
            # Aborted (self.sdr closed concurrently, or abort_check tripped)
            # before even one batch could be read — nothing to return. The
            # caller (_scan_cycle) already wraps capture_spectrum() in a
            # try/except that logs and retries, so raise rather than crash
            # on a None division below.
            raise RuntimeError("capture_spectrum aborted before any data was collected")

        psd_avg = psd_accum / batches_done

        # Convert to dB and shift to match frequency ordering
        power_db = 10 * np.log10(psd_avg)

        # Shift FFT output (fftshift equivalent)
        power_db = np.fft.fftshift(power_db)
        freqs = np.fft.fftshift(freqs) + self.center_freq
        
        return freqs, power_db
    
    def detect_signals(self, freqs: np.ndarray, power_db: np.ndarray,
                       threshold_db: float = None) -> List[DetectedSignal]:
        """
        Detect peaks in spectrum that could be radiosonde signals.

        threshold_db overrides self.detection_threshold for this call — used to
        run a second, lower-threshold pass for the frequency repository (list
        weak candidates) without lowering the decode threshold.

        Peak-picking approach follows radiosonde_auto_rx:
          - Noise floor = median of the *whole* power spectrum (not a
            low-percentile subset) — chosen there specifically for better
            outlier rejection than a mean.
          - Minimum peak-to-peak spacing (mpd) expressed in real Hz (channel
            spacing) and converted to FFT bins from the actual frequency
            resolution, instead of a fixed fraction of fft_size.
          - Surviving peaks are quantized to the channel-spacing grid,
            deduplicated (keep the strongest per bucket), sorted by strength,
            and capped to max_peaks so a busy band can't flood the decode
            pipeline with more candidates than can ever be serviced.
        """
        noise_floor = np.median(power_db)

        # Find peaks above threshold
        thr_db = self.detection_threshold if threshold_db is None else threshold_db
        threshold = noise_floor + thr_db

        self.logger.debug(f"Noise floor: {noise_floor:.1f} dB, Detection threshold: {threshold:.1f} dB")

        # Minimum distance between peaks, expressed in real Hz (channel
        # spacing) and converted to FFT bins via the actual frequency step.
        freq_step_hz = self.sample_rate / self.fft_size
        min_distance_bins = max(1, int(round(self.channel_spacing_hz / freq_step_hz)))

        # Use scipy peak detection
        peaks, properties = scipy_signal.find_peaks(
            power_db,
            height=threshold,
            distance=min_distance_bins,  # Minimum distance between peaks
            width=5  # Minimum width
        )

        if len(peaks) > 0 :
            self.logger.info(f"{len(peaks):d} peaks detected")

        detected = []
        for peak_idx in peaks:
            freq = freqs[peak_idx]
            strength = power_db[peak_idx] - noise_floor

            # Reject peaks at/near the tuner center frequency (DC/1-f spur —
            # see dc_notch_hz note in __init__)
            if self.dc_notch_hz > 0 and abs(freq - self.center_freq) < self.dc_notch_hz:
                self.logger.debug(
                    f"Rejected signal (DC spur region): {freq/1e6:.4f} MHz, "
                    f"{abs(freq - self.center_freq)/1e3:.1f} kHz from center"
                )
                continue
            
            # Estimate bandwidth (3dB bandwidth)
            half_power = power_db[peak_idx] - 3
            left_idx = peak_idx
            right_idx = peak_idx
            
            while left_idx > 0 and power_db[left_idx] > half_power:
                left_idx -= 1
            while right_idx < len(power_db) - 1 and power_db[right_idx] > half_power:
                right_idx += 1
            
            bandwidth = abs(freqs[right_idx] - freqs[left_idx])
            
            # Check blacklist (±2.5 kHz tolerance)
            is_blacklisted = any(abs(freq - bl) < 2_500 for bl in self.frequency_blacklist)
            if is_blacklisted:
                self.logger.debug(
                    f"Rejected signal (blacklisted): {freq/1e6:.4f} MHz, SNR: {strength:.1f} dB, BW: {bandwidth/1e3:.1f} kHz"
                )
                continue

            # Filter by radiosonde typical bandwidth (4-20 kHz)
            if not (1000 < bandwidth < 30000):
                self.logger.debug(
                    f"Rejected signal (bandwidth): {freq/1e6:.4f} MHz, BW: {bandwidth/1e3:.1f} kHz"
                )
                continue

            # Reject candidates that don't fall on the channel raster (see
            # check_frequency_raster in __init__). A strong signal's own
            # sidelobes can otherwise be mistaken for a second, off-channel
            # sonde a few kHz away.
            if self.check_frequency_raster and not self._is_on_raster(freq):
                self.logger.debug(
                    f"Rejected signal (off raster): {freq/1e6:.4f} MHz, SNR: {strength:.1f} dB, BW: {bandwidth/1e3:.1f} kHz"
                )
                continue

            self.logger.info(
                f"Found signal: {freq/1e6:.4f} MHz, SNR: {strength:.1f} dB, BW: {bandwidth/1e3:.1f} kHz"
            )
            detected.append(DetectedSignal(
                frequency=freq,
                strength=strength,
                bandwidth=bandwidth,
                timestamp=time.time(),
                power_dbfs=float(power_db[peak_idx]),
                noise_floor_dbfs=float(noise_floor)
            ))

        # Quantize to the real-world channel-spacing grid, deduplicate (keep
        # the strongest candidate per bucket — two peaks that both land in
        # the same 10 kHz channel after quantization are almost always the
        # same physical signal, not two distinct sondes), sort by strength,
        # and cap to max_peaks so a busy band can't overwhelm the decoders.
        quantized: Dict[float, DetectedSignal] = {}
        for sig in detected:
            bucket = round(sig.frequency / self.channel_spacing_hz) * self.channel_spacing_hz
            existing = quantized.get(bucket)
            if existing is None or sig.strength > existing.strength:
                quantized[bucket] = sig

        capped = sorted(quantized.values(), key=lambda s: s.strength, reverse=True)[:self.max_peaks]

        return capped

    def _is_on_raster(self, freq_hz: float) -> bool:
        """
        True if freq_hz falls within raster_tolerance_hz of a channel_spacing_hz
        grid point (e.g. a 10 kHz raster from 401.000-405.990 MHz). Handles
        drift in either direction from the nearest grid point, not just the
        one below it.
        """
        remainder = freq_hz % self.channel_spacing_hz
        if remainder > self.channel_spacing_hz / 2:
            remainder -= self.channel_spacing_hz
        return abs(remainder) <= self.raster_tolerance_hz

    def filter_signals_in_ranges(self, signals: List[DetectedSignal]) -> List[DetectedSignal]:
        """Filter signals to only those in configured frequency ranges"""
        freq_ranges = self.config['detection']['freq_ranges']
        filtered = []
        
        for sig in signals:
            for freq_min, freq_max in freq_ranges:
                if freq_min <= sig.frequency <= freq_max:
                    filtered.append(sig)
                    break
        
        return filtered
    
    def scan_spectrum(self):
        """Perform one spectrum scan and update detected signals"""
        try:
            freqs, power_db = self.capture_spectrum()
            detected = self.detect_signals(freqs, power_db)
            detected = self.filter_signals_in_ranges(detected)
            
            # Update detected signals list
            with self.lock:
                # Remove old detections (older than 30 seconds)
                current_time = time.time()
                self.detected_signals = [
                    s for s in self.detected_signals
                    if current_time - s.timestamp < 30
                ]
                
                # Add new detections (avoid duplicates within 10 kHz)
                for new_sig in detected:
                    duplicate = False
                    for existing_sig in self.detected_signals:
                        if abs(new_sig.frequency - existing_sig.frequency) < 10000:
                            # Update existing signal
                            existing_sig.strength = new_sig.strength
                            existing_sig.timestamp = new_sig.timestamp
                            duplicate = True
                            break
                    
                    if not duplicate:
                        self.detected_signals.append(new_sig)
                        self.logger.info(
                            f"New signal detected: {new_sig.frequency/1e6:.4f} MHz, "
                            f"SNR: {new_sig.strength:.1f} dB, "
                            f"BW: {new_sig.bandwidth/1e3:.1f} kHz"
                        )
            
        except Exception as e:
            self.logger.error(f"Error during spectrum scan: {e}", exc_info=True)
    
    def get_detected_signals(self) -> List[DetectedSignal]:
        """Get current list of detected signals"""
        with self.lock:
            return self.detected_signals.copy()
    
    def start_scanning(self):
        """Start continuous spectrum scanning in background thread"""
        if self.running:
            self.logger.warning("Spectrum scanning already running")
            return
        
        self.running = True
        self.scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self.scan_thread.start()
        self.logger.info("Spectrum scanning started")
    
    def stop_scanning(self):
        """Stop spectrum scanning"""
        self.running = False
        if hasattr(self, 'scan_thread'):
            self.scan_thread.join(timeout=5)
        self.logger.info("Spectrum scanning stopped")
    
    def pause(self):
        """
        Pause scanning and close RTL-SDR device
        This allows other processes (like rtl_fm) to access the device
        """
        if self.sdr:
            self.logger.info("Pausing spectrum analyzer - closing RTL-SDR device")
            try:
                self.sdr.close()
                self.sdr = None
            except Exception as e:
                self.logger.warning(f"Error closing RTL-SDR during pause: {e}")
    
    def resume(self):
        """
        Resume scanning by reopening RTL-SDR device
        """
        if not self.sdr:
            self.logger.info("Resuming spectrum analyzer - reopening RTL-SDR device")
            try:
                # Resolve serial to index the same way initialize() does
                if self.device_serial.isdigit():
                    device_index = int(self.device_serial)
                else:
                    device_index = RtlSdr.get_device_index_by_serial(self.device_serial)
                self.sdr = RtlSdr(device_index)
                
                # Reconfigure SDR from stored device config
                self.sdr.sample_rate = self.sample_rate
                self.sdr.center_freq = self.center_freq
                
                gain = self.device_config.get('gain', 0)
                if gain == 0:
                    self.sdr.gain = 'auto'
                else:
                    self.sdr.gain = gain
                    
                ppm = self.device_config.get('ppm_error', 0)
                if ppm != 0:
                    self.sdr.freq_correction = ppm
                
                self.logger.info("RTL-SDR reopened successfully")
            except Exception as e:
                self.logger.error(f"Failed to reopen RTL-SDR: {e}")
    
    def _scan_loop(self):
        """Background scanning loop"""
        while self.running:
            try:
                # Skip scanning if SDR is paused (closed)
                if self.sdr is not None:
                    self.scan_spectrum()
                time.sleep(self.scan_interval)
            except Exception as e:
                self.logger.error(f"Error in scan loop: {e}", exc_info=True)
                time.sleep(5)  # Wait before retrying
