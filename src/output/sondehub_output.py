"""
# =============================================================================
#  OpenWX -- Open Weather Radiosonde Telemetry System
# =============================================================================
#
#  File   : sondehub_output.py
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
#  Direct SondeHub v2 telemetry upload plugin for OpenWX.
#
#  Uploads individual decoded radiosonde telemetry frames to the SondeHub
#  amateur radiosonde tracking network. Each frame is dispatched in its own
#  background thread with per-serial rate limiting to avoid overloading the
#  API. For high-volume or unstable-link scenarios use sondehub_queue.py.
#
#  Upload pipeline:
#    send_telemetry() → rate-limit check → daemon thread
#      → gzip-compressed JSON array → PUT api.v2.sondehub.org/sondes/telemetry
#
#  Features: per-serial subtype/satellite continuity, jittered exponential
#  retry with Retry-After support, periodic listener metadata registration,
#  strict serial format validation (RS41, RS92, DFM, M10, iMet, ...).
#
#  Dependency: requests
#  - JSON array payload (gzip compressed)
#  - User-Agent header
#  - RFC7231 Date header
#  - Content-Encoding: gzip header
#
# =============================================================================
"""

import gzip
import importlib
import json
import logging
import threading
import time
import random
from datetime import datetime, timezone
from email.utils import formatdate
from typing import Dict, Optional

from .. import __software_name__, __version__
from ..decoders.models import SondeTelemetry


class SondeHubOutput:
    """Uploads radiosonde telemetry frames to SondeHub v2."""

    DEFAULT_UPLOAD_URL = 'https://api.v2.sondehub.org/sondes/telemetry'
    DEFAULT_LISTENERS_URL = 'https://api.v2.sondehub.org/listeners'

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger('SondeHubOutput')

        sh_cfg = config.get('sondehub', {})
        self.enabled = bool(sh_cfg.get('enabled', False))
        self.upload_url = sh_cfg.get('upload_url', self.DEFAULT_UPLOAD_URL)
        self.listeners_url = sh_cfg.get('listeners_url', self.DEFAULT_LISTENERS_URL)
        self.upload_rate_s = max(1, int(sh_cfg.get('upload_rate_s', 15)))
        self.listener_upload_interval_s = max(60, int(sh_cfg.get('listener_upload_interval_s', 900)))

        # Always use internal application identity/version for uploads.
        self.software_name = __software_name__
        self.software_version = __version__

        station_cfg = config.get('station', {})
        self.uploader_callsign = (
            sh_cfg.get('uploader_callsign')
            or sh_cfg.get('station_id')
            or station_cfg.get('callsign')
            or 'OPENWXSDR_STATION'
        )
        self.uploader_antenna = sh_cfg.get('uploader_antenna', '')
        self.uploader_radio = sh_cfg.get('uploader_radio', '')
        self.contact_email = sh_cfg.get('contact_email', '')

        self.station_lat = sh_cfg.get('uploader_lat', station_cfg.get('lat'))
        self.station_lon = sh_cfg.get('uploader_lon', station_cfg.get('lon'))
        self.station_alt = sh_cfg.get('uploader_alt', station_cfg.get('alt'))

        self._last_upload_t: Dict[str, float] = {}
        self._last_subtype_by_serial: Dict[str, str] = {}
        self._last_sats_by_serial: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._last_listener_upload_t = 0.0  # Will force upload on first telemetry
        self._session = None

        self.logger.debug(f"[SONDEHUB-INIT] enabled={self.enabled}, callsign={self.uploader_callsign}, upload_url={self.upload_url}")

        if self.enabled:
            try:
                requests_module = importlib.import_module('requests')
                self._session = requests_module.Session()
            except ImportError:
                self.logger.error("requests library not installed; SondeHub upload disabled")
                self.enabled = False
                return
            self.logger.info(
                f"SondeHub upload enabled -> {self.upload_url} "
                f"(callsign={self.uploader_callsign}, rate={self.upload_rate_s}s, "
                f"listener_rate={self.listener_upload_interval_s}s)"
            )
            # Queue immediate listener metadata upload on startup
            timer = threading.Timer(0.5, self._upload_listener_metadata)
            timer.daemon = True
            timer.start()
        else:
            self.logger.debug("SondeHub upload is disabled in configuration")

    def send_telemetry(self, telemetry: SondeTelemetry):
        """Queue a telemetry frame for SondeHub upload (non-blocking)."""
        if not self.enabled:
            self.logger.debug("SondeHub disabled; skipping send_telemetry")
            return

        self.logger.info(
            f"[SONDEHUB] Queue telemetry serial={telemetry.serial} "
            f"frame={telemetry.frame_number} raw_sats={getattr(telemetry, 'satellites', None)}"
        )

        self.logger.debug(
            f"[SONDEHUB] Received telemetry: serial={telemetry.serial}, "
            f"type={telemetry.sonde_type}, frame={telemetry.frame_number}, "
            f"has_position={telemetry.position is not None}"
        )

        # Keep upload volume manageable while preserving fresh updates.
        key = telemetry.serial or 'UNKNOWN'
        now = time.monotonic()
        with self._lock:
            last_t = self._last_upload_t.get(key, 0.0)
            time_since_last = now - last_t
            if time_since_last < self.upload_rate_s:
                self.logger.debug(
                    f"[SONDEHUB] Rate limiting {key}: {time_since_last:.1f}s < {self.upload_rate_s}s, skipping"
                )
                return
            self._last_upload_t[key] = now

        self.logger.debug(f"[SONDEHUB] Queuing upload for {key}")
        t = threading.Thread(
            target=self._upload,
            args=(telemetry,),
            daemon=True,
            name=f"SondeHubUpload-{key}",
        )
        t.start()

    def close(self):
        """No persistent connection currently used."""
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass

    def _retry_delay_s(self, retries: int, resp_headers: Optional[dict] = None) -> float:
        """Compute retry delay with Retry-After support and jittered backoff."""
        if resp_headers:
            retry_after = resp_headers.get('Retry-After')
            if retry_after:
                try:
                    return max(0.25, float(retry_after))
                except (TypeError, ValueError):
                    pass

        base = min(5.0, 0.5 * (2 ** max(0, retries - 1)))
        return base + random.uniform(0.0, 0.25)

    def get_status(self) -> dict:
        """Return SondeHub upload status for health endpoints."""
        if not self.enabled:
            return {'status': 'disabled'}

        with self._lock:
            has_uploaded = bool(self._last_upload_t)

        return {
            'status': 'active' if has_uploaded else 'waiting',
            'upload_rate_s': self.upload_rate_s,
            'callsign': self.uploader_callsign,
        }

    def _utc_iso(self, dt: Optional[datetime]) -> str:
        if dt is None:
            dt = datetime.now(timezone.utc)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat(timespec='milliseconds').replace('+00:00', 'Z')

    def _manufacturer_for(self, sonde_type: str) -> str:
        st = (sonde_type or '').strip()
        base = st.split('-', 1)[0].upper() if st else ''
        mapping = {
            'RS41': 'Vaisala',
            'RS92': 'Vaisala',
            'DFM': 'Graw',
            'M10': 'Meteomodem',
            'M20': 'Meteomodem',
            'IMET': 'Intermet Systems',
            'LMS6': 'Lockheed Martin',
            'MRZ': 'Meteo-Radiy',
        }
        return mapping.get(base, 'Unknown')

    def _effective_type(self, telemetry: SondeTelemetry) -> str:
        """Return normalized base sonde type (e.g. RS41 from RS41-SGP)."""
        sonde_type = (telemetry.sonde_type or '').strip() or 'Unknown'
        subtype = (telemetry.subtype or '').strip()

        if '-' in sonde_type:
            return sonde_type.split('-', 1)[0].upper()
        if '-' in subtype:
            return subtype.split('-', 1)[0].upper()
        return sonde_type

    def _effective_subtype(self, telemetry: SondeTelemetry, serial: str) -> str:
        """Return best subtype for SondeHub, with per-serial continuity — but
        only for sonde types where radiosonde_auto_rx's own uploader
        (sondehub.py reformat_data) would upload a subtype at all.

        Previously this passed through whatever a decoder put in
        telemetry.subtype for ANY sonde type. A SondeHub maintainer flagged
        that e.g. the M20 decoder emits a 'subtype' field that isn't
        meaningful for uploads and can cause weird tracker behaviour — auto_rx
        never uploads a subtype for M10/M20 at all. Mirror auto_rx's per-type
        policy (same fix already applied in sondehub_queue.py) instead of a
        blanket "if present, upload it" rule.
        """
        raw_subtype = (telemetry.subtype or '').strip()
        sonde_type = (telemetry.sonde_type or '').strip()
        base = self._effective_type(telemetry).upper()

        if base not in ('RS41', 'RS92', 'DFM', 'LMS6', 'MRZ', 'WXR301'):
            # auto_rx never uploads a subtype for these types (e.g. M10/M20) —
            # omit it rather than passing through whatever the decoder produced.
            return ''

        if base == 'WXR301':
            # auto_rx only sets a subtype for the PN9 variant, nothing else.
            with self._lock:
                if raw_subtype == 'WXR_PN9':
                    self._last_subtype_by_serial[serial] = 'WxR-301D-5k'
                    return 'WxR-301D-5k'
                return self._last_subtype_by_serial.get(serial, '')

        subtype = raw_subtype

        # If decoder only provides RS41-SGP as type, map it into subtype too.
        if not subtype and '-' in sonde_type:
            subtype = sonde_type

        # Heuristic for RS41 where decoder doesn't emit subtype on every line.
        # V-prefixed serials are commonly RS41-SGP.
        if not subtype and base == 'RS41' and serial.upper().startswith('V'):
            subtype = 'RS41-SGP'

        # Some decoders only emit base type for RS41; still provide subtype for consistency.
        if not subtype and base == 'RS41':
            subtype = 'RS41'

        with self._lock:
            if subtype:
                self._last_subtype_by_serial[serial] = subtype
                return subtype
            return self._last_subtype_by_serial.get(serial, '')

    def _normalize_sats(self, value) -> Optional[int]:
        """Normalize a satellite count candidate into a non-negative int."""
        if value is None:
            return None
        try:
            sats_i = int(value)
        except (TypeError, ValueError):
            return None
        return sats_i if sats_i >= 0 else None

    def _effective_sats(self, telemetry: SondeTelemetry, serial: str) -> Optional[int]:
        """Return sats with per-serial continuity when current frame omits it."""
        candidate_values = [
            getattr(telemetry, 'satellites', None),
            getattr(telemetry, 'sats', None),
        ]

        try:
            candidate_values.append(telemetry.to_dict().get('sats'))
        except Exception:
            pass

        for candidate in candidate_values:
            sats_i = self._normalize_sats(candidate)
            if sats_i is not None:
                with self._lock:
                    self._last_sats_by_serial[serial] = sats_i
                return sats_i

        with self._lock:
            return self._last_sats_by_serial.get(serial)

    def _uploader_position(self) -> Optional[list]:
        """Return uploader position as [lat, lon, alt] for SondeHub."""
        if self.station_lat is None or self.station_lon is None:
            return None

        alt = float(self.station_alt) if self.station_alt is not None else 0.0
        return [
            round(float(self.station_lat), 6),
            round(float(self.station_lon), 6),
            round(alt, 1),
        ]

    def _is_valid_serial(self, serial: str, sonde_type: str) -> bool:
        """
        Validate sonde serial format according to SondeHub requirements.
        
        RS41/RS92: Must start with A-Z followed by 7-8 digits (e.g., V1220530, S12345678)
        DFM: Must start with 'D' followed by 8 digits (e.g., D12345678)
        M10/M20: Must start with 'M' followed by 8-10 characters
        iMet: Must start with 'iMet' or 'IMET'
        LMS6: Starts with 'LMS'
        MRZ: Starts with 'MRZ'
        
        Returns False for partial/malformed serials like '-+', 'UNKNOWN', etc.
        """
        import re
        
        if not serial or serial == 'UNKNOWN':
            return False
        
        serial = serial.strip()
        sonde_type_upper = sonde_type.upper().split('-')[0]  # Strip subtype like RS41-SGP → RS41
        
        # RS41/RS92: [A-Z][0-9]{7,8} (8 or 9 chars total, e.g., V1220530 or S12345678)
        if sonde_type_upper in ('RS41', 'RS92'):
            return bool(re.match(r'^[A-Z][0-9]{7,8}$', serial))
        
        # DFM: Accept either D[0-9]{8} (with prefix) or [0-9]{8} (JSON format without prefix)
        # Examples: "D21062636" or "21062636"
        elif sonde_type_upper == 'DFM':
            return bool(re.match(r'^(D)?[0-9]{6,8}$', serial))
        
        # M10/M20: M[0-9A-Z]{8,10}
        elif sonde_type_upper in ('M10', 'M20'):
            return bool(re.match(r'^M[0-9A-Z]{8,10}$', serial, re.IGNORECASE))
        
        # iMet: Starts with iMet or IMET
        elif sonde_type_upper == 'IMET':
            return serial.upper().startswith('IMET') and len(serial) >= 4
        
        # LMS6: Starts with LMS
        elif sonde_type_upper == 'LMS6':
            return serial.upper().startswith('LMS') and len(serial) >= 4
        
        # MRZ: Starts with MRZ
        elif sonde_type_upper == 'MRZ':
            return serial.upper().startswith('MRZ') and len(serial) >= 4
        
        # Unknown type: reject if it contains common malformed patterns
        # Reject serials with only special characters, spaces, or very short
        if len(serial) < 3:
            return False
        if re.match(r'^[-+\s]+$', serial):
            return False
        
        # Allow other types with alphanumeric serials
        return bool(re.match(r'^[A-Z0-9][A-Z0-9\-]{2,}$', serial, re.IGNORECASE))

    def _build_payload(self, telemetry: SondeTelemetry) -> Optional[dict]:
        if not telemetry.position:
            self.logger.debug(f"[SONDEHUB] No position data for {telemetry.serial}, skipping payload")
            return None

        serial = (telemetry.serial or '').strip()
        if not serial or serial == 'UNKNOWN':
            self.logger.debug(f"[SONDEHUB] Invalid serial {serial}, skipping payload")
            return None
        
        # Filter out invalid frame numbers (0 or None = incomplete decode)
        if not telemetry.frame_number or telemetry.frame_number == 0:
            self.logger.warning(
                f"[SONDEHUB] Invalid frame_number={telemetry.frame_number} for {serial}, "
                f"skipping upload (frame not successfully decoded)"
            )
            return None

        sonde_type = self._effective_type(telemetry)
        
        # Validate serial format before building payload
        if not self._is_valid_serial(serial, sonde_type):
            self.logger.warning(
                f"[SONDEHUB] Invalid serial format: '{serial}' for {sonde_type}. "
                f"Skipping upload (likely partial/corrupted decode). "
                f"Valid formats: RS41=[A-Z][0-9]{{7-8}}, DFM=D[0-9]{{8}}"
            )
            return None
        
        subtype = self._effective_subtype(telemetry, serial)
        sats = self._effective_sats(telemetry, serial)

        self.logger.info(
            f"[SONDEHUB] Build payload serial={serial} "
            f"raw_sats={getattr(telemetry, 'satellites', None)} resolved_sats={sats}"
        )

        # SondeHub guidance: avoid uploading DFM frames before serial is known.
        if sonde_type.upper().startswith('DFM') and serial in ('UNKNOWN', ''):
            self.logger.debug(f"[SONDEHUB] Skipping DFM frame without known serial")
            return None
        
        self.logger.debug(f"[SONDEHUB] Building payload for {serial} ({sonde_type})")

        payload = {
            'software_name': self.software_name,
            'software_version': self.software_version,
            'uploader_callsign': self.uploader_callsign,
            'time_received': self._utc_iso(datetime.now(timezone.utc)),
            'manufacturer': self._manufacturer_for(sonde_type),
            'type': sonde_type,
            'serial': serial,
            'frame': int(telemetry.frame_number or 0),
            'datetime': self._utc_iso(telemetry.position.datetime),
            'lat': round(float(telemetry.position.latitude), 6),
            'lon': round(float(telemetry.position.longitude), 6),
            'alt': round(float(telemetry.position.altitude), 1),
        }

        if subtype:
            payload['subtype'] = subtype

        if sats is not None:
            payload['sats'] = sats

        if telemetry.frequency:
            payload['frequency'] = round(float(telemetry.frequency) / 1e6, 3)

        # CRITICAL: decoders use sentinel values to mean "no data" (e.g. -9999.0
        # for velocity/heading, -273.0 for temp, -1.0 for humidity/pressure/batt —
        # see DECODER_OPTIONAL_FIELDS defaults in the reference auto_rx decoder).
        # auto_rx's own uploader explicitly checks against these sentinels before
        # including a field; we previously only checked "is not None" (or nothing
        # at all for heading), which lets literal sentinel garbage through as if
        # it were a real reading.
        if telemetry.velocity:
            vel_h = telemetry.velocity.horizontal_speed
            vel_v = telemetry.velocity.vertical_speed
            heading = telemetry.velocity.heading
            if vel_h is not None and vel_h > -9999.0:
                payload['vel_h'] = round(float(vel_h), 2)
            if vel_v is not None and vel_v > -9999.0:
                payload['vel_v'] = round(float(vel_v), 2)
            if heading is not None and heading > -9999.0:
                payload['heading'] = round(float(heading), 1)

        if telemetry.environment:
            temp = telemetry.environment.temperature
            humidity = telemetry.environment.humidity
            pressure = telemetry.environment.pressure
            if temp is not None and temp > -273.0:
                payload['temp'] = round(float(temp), 1)
            if humidity is not None and humidity >= 0.0:
                payload['humidity'] = round(float(humidity), 1)
            if pressure is not None and pressure >= 0.0:
                payload['pressure'] = round(float(pressure), 2)
            self.logger.debug(
                f"[SONDEHUB-PTU] {serial}: Added environment to payload: "
                f"temp={payload.get('temp')}, hum={payload.get('humidity')}, pres={payload.get('pressure')}"
            )
        else:
            self.logger.debug(f"[SONDEHUB-PTU] {serial}: No environment data in telemetry object")

        if telemetry.battery is not None and telemetry.battery >= 0.0:
            payload['batt'] = round(float(telemetry.battery), 2)

        # RS41-specific fields
        if telemetry.burst_timer is not None:
            payload['burst_timer'] = int(telemetry.burst_timer)
        if telemetry.rs41_mainboard is not None:
            payload['rs41_mainboard'] = str(telemetry.rs41_mainboard)
        if telemetry.rs41_mainboard_fw is not None:
            # SondeHub expects rs41_mainboard_fw as a STRING, not an int.
            payload['rs41_mainboard_fw'] = str(telemetry.rs41_mainboard_fw)
        if telemetry.ref_datetime is not None:
            payload['ref_datetime'] = str(telemetry.ref_datetime)
        if telemetry.ref_position is not None:
            payload['ref_position'] = str(telemetry.ref_position)
        if telemetry.tx_frequency is not None:
            # tx_frequency is stored in kHz (field-observed: 405700.0 = the
            # nominal 405.7 MHz channel). Convert kHz→MHz so it's submitted as
            # 405.7. (Dividing by 1e6 as if it were Hz gave a wrong 0.406.)
            payload['tx_frequency'] = round(float(telemetry.tx_frequency) / 1e3, 3)

        # SNR and RSSI removed per user request - not needed for SondeHub

        uploader_position = self._uploader_position()
        if uploader_position is not None:
            payload['uploader_position'] = uploader_position
        if self.station_alt is not None:
            payload['uploader_alt'] = float(self.station_alt)
        if self.uploader_antenna:
            payload['uploader_antenna'] = self.uploader_antenna

        self.logger.debug(
            f"[SONDEHUB] Payload built: serial={payload.get('serial')}, "
            f"temp={payload.get('temp', 'N/A')}, "
            f"lat={payload.get('lat')}, lon={payload.get('lon')}"
        )
        return payload

    def _headers(self, with_gzip: bool = False) -> dict:
        headers = {
            'User-Agent': f"{self.software_name}-{self.software_version}",
            'Date': formatdate(timeval=None, localtime=False, usegmt=True),
            'Content-Type': 'application/json',
        }
        if with_gzip:
            headers['Content-Encoding'] = 'gzip'
        return headers

    def _upload_listener_metadata(self):
        """Optionally register listener metadata so station can show on map."""
        if not self.enabled:
            return

        now = time.monotonic()
        # Refresh occasionally, not every frame. Throttle by attempt time,
        # so repeated failures do not cause request floods.
        with self._lock:
            if (now - self._last_listener_upload_t) < self.listener_upload_interval_s:
                self.logger.debug(
                    f"Listener metadata upload throttled ({self.listener_upload_interval_s}s rate limit)"
                )
                return
            self._last_listener_upload_t = now

        self.logger.debug("Preparing listener metadata upload")

        listener = {
            'software_name': self.software_name,
            'software_version': self.software_version,
            'uploader_callsign': self.uploader_callsign,
            'mobile': False,
        }
        uploader_position = self._uploader_position()
        if uploader_position is not None:
            listener['uploader_position'] = uploader_position
        if self.uploader_antenna:
            listener['uploader_antenna'] = self.uploader_antenna
        if self.contact_email:
            listener['uploader_contact_email'] = self.contact_email
        
        self.logger.debug(f"[SONDEHUB] Listener object built: callsign={listener.get('uploader_callsign')}, has_position={('uploader_position' in listener)}, antenna={listener.get('uploader_antenna', 'N/A')}")

        try:
            self.logger.debug(f"[SONDEHUB] Listener payload: {listener}")

            # Upload with retries for 500 errors
            retries = 0
            max_retries = 3
            upload_ok = False

            while retries < max_retries:
                try:
                    client = self._session
                    if client is None:
                        client = importlib.import_module('requests')

                    resp = client.put(
                        self.listeners_url,
                        json=listener,
                        headers=self._headers(),
                        timeout=(10, 6.1),
                    )

                    if 200 <= resp.status_code < 300:
                        self.logger.info(
                            f"[SONDEHUB✓] Listener metadata uploaded successfully to {self.listeners_url} "
                            f"for {listener.get('uploader_callsign')} (HTTP {resp.status_code})"
                        )
                        upload_ok = True
                        break
                    elif resp.status_code in (429, 500, 502, 503, 504):
                        retries += 1
                        if retries < max_retries:
                            delay_s = self._retry_delay_s(retries, resp.headers)
                            self.logger.debug(
                                f"[SONDEHUB] Listener upload transient HTTP {resp.status_code}, "
                                f"retrying in {delay_s:.2f}s ({retries}/{max_retries})"
                            )
                            time.sleep(delay_s)
                            continue
                        else:
                            self.logger.warning(
                                f"[SONDEHUB✗] Listener upload failed after {max_retries} retries: HTTP {resp.status_code}"
                            )
                            break
                    else:
                        self.logger.warning(
                            f"[SONDEHUB✗] Listener upload failed: HTTP {resp.status_code} | {resp.text[:200]}"
                        )
                        break
                except Exception as exc:
                    retries += 1
                    if retries < max_retries:
                        delay_s = self._retry_delay_s(retries)
                        self.logger.debug(
                            f"[SONDEHUB] Listener request error: {exc}, retrying in {delay_s:.2f}s "
                            f"({retries}/{max_retries})"
                        )
                        time.sleep(delay_s)
                    else:
                        raise

            if not upload_ok:
                self.logger.debug(f"[SONDEHUB] Listener payload was: {listener}")
        except Exception as exc:
            self.logger.error(
                f"[SONDEHUB✗] Listener upload error: {type(exc).__name__}: {exc} "
                f"| URL: {self.listeners_url} | Callsign: {listener.get('uploader_callsign')}"
            )

    def _upload(self, telemetry: SondeTelemetry):
        self.logger.debug(f"[SONDEHUB] _upload thread starting for {telemetry.serial}")
        try:
            payload = self._build_payload(telemetry)
            if not payload:
                self.logger.debug(f"[SONDEHUB] Failed to build payload for {telemetry.serial}")
                return

            # Endpoint expects an array of telemetry objects.
            self.logger.info(
                f"[SONDEHUB] Uploading telemetry to {self.upload_url} "
                f"for {telemetry.serial} "
                f"(frame={payload.get('frame')}, "
                f"manufacturer={payload.get('manufacturer')}, "
                f"type={payload.get('type')}, "
                f"subtype={payload.get('subtype', 'n/a')}, "
                f"sats={payload.get('sats', 'n/a')})"
            )

            # Compress the JSON payload
            telem_json = json.dumps([payload]).encode('utf-8')
            compressed_payload = gzip.compress(telem_json)
            self.logger.debug(
                f"[SONDEHUB] Payload compression: {len(telem_json)} bytes -> {len(compressed_payload)} bytes "
                f"({100*len(compressed_payload)/len(telem_json):.1f}% ratio)"
            )

            # Upload with retries for 500 errors
            retries = 0
            max_retries = 3
            upload_ok = False

            while retries < max_retries:
                try:
                    client = self._session
                    if client is None:
                        client = importlib.import_module('requests')

                    resp = client.put(
                        self.upload_url,
                        compressed_payload,
                        headers=self._headers(with_gzip=True),
                        timeout=(10, 6.1),
                    )

                    if 200 <= resp.status_code < 300:
                        # 202 means there were warnings/errors in the data, but still accepted
                        try:
                            resp_json = resp.json()
                            for error in resp_json.get('errors', []):
                                msg = error.get('error_message', 'unknown')
                                if 'z-check' not in msg:
                                    self.logger.warning(f"[SONDEHUB] Data error: {msg}")
                                else:
                                    self.logger.debug(f"[SONDEHUB] Data error: {msg}")
                            for warning in resp_json.get('warnings', []):
                                self.logger.debug(f"[SONDEHUB] Data warning: {warning.get('warning_message', 'unknown')}")
                        except Exception:
                            pass
                        self.logger.info(
                            f"[SONDEHUB✓] Upload OK for {telemetry.serial}: HTTP {resp.status_code}"
                        )
                        upload_ok = True
                        break
                    elif resp.status_code in (429, 500, 502, 503, 504):
                        retries += 1
                        if retries < max_retries:
                            delay_s = self._retry_delay_s(retries, resp.headers)
                            self.logger.debug(
                                f"[SONDEHUB] Transient HTTP {resp.status_code}, retrying in {delay_s:.2f}s "
                                f"({retries}/{max_retries})"
                            )
                            time.sleep(delay_s)
                            continue
                        else:
                            self.logger.warning(
                                f"[SONDEHUB✗] Upload failed after {max_retries} retries: HTTP {resp.status_code}"
                            )
                            break
                    else:
                        self.logger.warning(
                            f"[SONDEHUB✗] Upload failed for {telemetry.serial}: "
                            f"HTTP {resp.status_code} | {resp.text[:200]}"
                        )
                        break
                except Exception as exc:
                    retries += 1
                    if retries < max_retries:
                        delay_s = self._retry_delay_s(retries)
                        self.logger.debug(
                            f"[SONDEHUB] Request error: {exc}, retrying in {delay_s:.2f}s "
                            f"({retries}/{max_retries})"
                        )
                        time.sleep(delay_s)
                    else:
                        raise

            if upload_ok:
                self._upload_listener_metadata()
        except Exception as exc:
            self.logger.error(
                f"[SONDEHUB✗] Upload exception for {telemetry.serial}: "
                f"{type(exc).__name__}: {exc}"
            )
