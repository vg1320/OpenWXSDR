"""
# =============================================================================
#  OpenWX -- Open Weather Radiosonde Telemetry System
# =============================================================================
#
#  File   : signal_metrics.py
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
#  Rolling signal metrics (RSSI/SNR-like) computed from live IQ samples.
#
#  Provides SignalMetrics, a thread-safe rolling IQ power estimator that
#  computes RSSI (dBFS and gain-compensated dBm) and SNR from a sliding
#  window of instantaneous IQ power samples.
#
#  RSSI reported as estimated dBm using a gain-compensated dBFS mapping:
#    rssi_dbm ≈ rssi_dbfs − gain_db + calibration_db
#
#  SNR is derived from the difference between instantaneous signal power
#  and the 20th-percentile noise floor of the rolling window.
#
#  Used by : AudioPipeline, AirspyPipeline, AirspyChannelizer
#
# =============================================================================
"""

from collections import deque
from dataclasses import dataclass, field
import math
import threading
import time
from typing import Deque, Optional, Tuple

import numpy as np


@dataclass
class SignalMetrics:
    """Thread-safe rolling IQ power/SNR estimator."""

    window_size: int = 200
    gain_db: float = 0.0
    calibration_db: float = -14.0
    power_hist: Deque[float] = field(default_factory=lambda: deque(maxlen=200))
    latest_rssi_dbfs: float = -140.0
    latest_rssi_dbm: float = -140.0
    latest_snr_db: float = 0.0
    # Recompute at most every min_interval_s. Pumps call update_iq once per audio
    # chunk — hundreds of times/s on the 10 MSPS Airspy channelizer — but per-frame
    # RSSI/SNR only needs ~1/s, so throttling to ~10/s cuts the metric CPU ~40x
    # with no visible loss. Set 0.0 to disable throttling.
    min_interval_s: float = 0.1

    def __post_init__(self):
        # Recreate deque to honor custom window_size when explicitly passed.
        self.power_hist = deque(self.power_hist, maxlen=max(20, int(self.window_size)))
        self._lock = threading.Lock()
        self._last_update = 0.0

    def update_iq(self, i_samples: np.ndarray, q_samples: np.ndarray) -> None:
        """Update rolling metrics from float IQ vectors (throttled to
        min_interval_s so a high-rate pump doesn't burn CPU on redundant recomputes)."""
        if i_samples is None or q_samples is None:
            return
        if len(i_samples) == 0 or len(q_samples) == 0:
            return
        if self.min_interval_s > 0.0:
            now = time.monotonic()
            if now - self._last_update < self.min_interval_s:
                return
            self._last_update = now

        i = i_samples.astype(np.float32, copy=False)
        q = q_samples.astype(np.float32, copy=False)

        # Mean instantaneous IQ power in linear scale.
        p = np.mean(i * i + q * q)
        p = max(float(p), 1e-12)

        with self._lock:
            self.power_hist.append(p)
            self.latest_rssi_dbfs = 10.0 * math.log10(p)
            self.latest_rssi_dbm = self.latest_rssi_dbfs - float(self.gain_db) + float(self.calibration_db)

            if len(self.power_hist) >= 10:
                noise_p = float(np.percentile(np.array(self.power_hist, dtype=np.float32), 20.0))
            else:
                noise_p = p
            noise_p = max(noise_p, 1e-12)
            noise_db = 10.0 * math.log10(noise_p)
            self.latest_snr_db = max(0.0, self.latest_rssi_dbfs - noise_db)

    def snapshot(self) -> Tuple[Optional[float], Optional[float]]:
        with self._lock:
            return self.latest_rssi_dbm, self.latest_snr_db
