from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple

import cv2
import numpy as np

from config.constants import (
    RHYTHM_WINDOW_FRAMES,
    SYNC_CORRELATION_MIN,
    SYNC_PHASE_TOLERANCE_FRAMES,
)


class RhythmAnalyzer:

    def __init__(self, window_frames: int = RHYTHM_WINDOW_FRAMES,
                 sync_phase_tolerance: int = SYNC_PHASE_TOLERANCE_FRAMES,
                 sync_corr_min: float = SYNC_CORRELATION_MIN) -> None:
        self._window = window_frames
        self._phase_tol = sync_phase_tolerance
        self._corr_min = sync_corr_min
        self._signals: Dict[str, deque] = {}

    def update(self, channel_id: str, vu_roi: np.ndarray) -> None:
        """Acumula a luminosidade média do VU meter como sinal temporal do canal."""
        if channel_id not in self._signals:
            self._signals[channel_id] = deque(maxlen=self._window)
        self._signals[channel_id].append(self._mean_brightness(vu_roi))

    def are_in_sync(self, ch_a: str, ch_b: str) -> Tuple[bool, float]:
        """Compara ritmos de dois canais via cross-correlação dos sinais de VU meter.

        Retorna (in_sync, phase_offset_frames).
        """
        if ch_a not in self._signals or ch_b not in self._signals:
            return True, 0.0

        sig_a = np.array(self._signals[ch_a], dtype=float)
        sig_b = np.array(self._signals[ch_b], dtype=float)

        if len(sig_a) < 10 or len(sig_b) < 10:
            return True, 0.0

        min_len = min(len(sig_a), len(sig_b))
        sig_a, sig_b = sig_a[-min_len:] - sig_a.mean(), sig_b[-min_len:] - sig_b.mean()

        std_product = sig_a.std() * sig_b.std()
        if std_product < 1e-6:
            return True, 0.0

        corr = np.correlate(sig_a, sig_b, mode="full")
        norm_corr = corr / (std_product * min_len)
        peak_idx = int(np.argmax(norm_corr))
        phase_offset = float(peak_idx - (min_len - 1))
        max_corr = float(norm_corr[peak_idx])

        in_sync = abs(phase_offset) <= self._phase_tol and max_corr >= self._corr_min
        return in_sync, phase_offset

    def get_signal(self, channel_id: str) -> List[float]:
        """Retorna o sinal acumulado de luminosidade do canal."""
        return list(self._signals.get(channel_id, []))

    @staticmethod
    def _mean_brightness(roi: np.ndarray) -> float:
        """Converte ROI para escala de cinza e retorna luminosidade média."""
        if roi is None or roi.size == 0:
            return 0.0
        return float(np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)))
