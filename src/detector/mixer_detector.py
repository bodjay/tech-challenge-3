from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

from config.constants import (
    CLIPPING_HUE_HIGH_1, CLIPPING_HUE_HIGH_2,
    CLIPPING_HUE_LOW_1, CLIPPING_HUE_LOW_2,
    CLIPPING_RED_RATIO_THRESHOLD,
    CLIPPING_SAT_MIN, CLIPPING_VAL_MIN,
    FADER_ACTIVE_MIN_POSITION,
)


@dataclass
class ChannelState:
    id: str
    fader_position: float
    is_active: bool
    is_clipping: bool


@dataclass
class MixerPhysicalState:
    channels: List[ChannelState]
    master: ChannelState
    eq_active: bool
    crossfader_moving: bool
    faders_moving: bool
    jog_active: bool


EQ_MOTION_THRESHOLD = 0.04


class MixerDetector:

    def __init__(self, channels_config: List[dict], master_config: dict,
                 fader_active_min: float = FADER_ACTIVE_MIN_POSITION,
                 clip_ratio_threshold: float = CLIPPING_RED_RATIO_THRESHOLD,
                 zones: dict = None) -> None:
        self._channels_config = channels_config
        self._master_config = master_config
        self._fader_active_min = fader_active_min
        self._clip_ratio_threshold = clip_ratio_threshold
        self._zones = zones or {}
        self._prev_eq_gray: np.ndarray | None = None
        self._prev_crossfader_gray: np.ndarray | None = None
        self._prev_jog_grays: Dict[str, np.ndarray] = {}
        self._prev_fader_positions: Dict[str, float] = {}

    def detect(self, frame: np.ndarray) -> Tuple[List[ChannelState], ChannelState]:
        """Retorna o estado de cada canal e do master a partir das ROIs configuradas."""
        channels = [self._analyze_channel(frame, cfg) for cfg in self._channels_config]
        master = self._analyze_channel(frame, self._master_config)
        return channels, master

    def detect_physical(self, frame: np.ndarray) -> MixerPhysicalState:
        """Detecta o estado físico completo do mixer incluindo movimentos de EQ, jog e crossfader."""
        channels, master = self.detect(frame)
        return MixerPhysicalState(
            channels=channels,
            master=master,
            eq_active=self._detect_roi_motion(frame, self._zones.get("eq_knobs"), "_prev_eq_gray"),
            crossfader_moving=self._detect_roi_motion(frame, self._zones.get("crossfader"), "_prev_crossfader_gray"),
            faders_moving=self._detect_faders_moving(channels),
            jog_active=self._detect_jog_motion(frame),
        )

    def _detect_roi_motion(self, frame: np.ndarray, roi, attr: str) -> bool:
        if roi is None:
            return False
        crop = self._crop(frame, roi)
        if crop is None:
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        prev = getattr(self, attr)
        setattr(self, attr, gray.copy())
        if prev is None or prev.shape != gray.shape:
            return False
        diff = cv2.absdiff(gray, prev)
        return float(np.count_nonzero(diff > 15)) / diff.size > EQ_MOTION_THRESHOLD

    def _detect_jog_motion(self, frame: np.ndarray) -> bool:
        for cfg in self._channels_config:
            roi = cfg.get("jog_roi")
            if not roi:
                continue
            ch_id = str(cfg.get("id", ""))
            crop = self._crop(frame, roi)
            if crop is None:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            prev = self._prev_jog_grays.get(ch_id)
            self._prev_jog_grays[ch_id] = gray.copy()
            if prev is None or prev.shape != gray.shape:
                continue
            diff = cv2.absdiff(gray, prev)
            if float(np.count_nonzero(diff > 15)) / diff.size > EQ_MOTION_THRESHOLD:
                return True
        return False

    def _detect_faders_moving(self, channels: List[ChannelState]) -> bool:
        moving = False
        for ch in channels:
            prev = self._prev_fader_positions.get(ch.id)
            if prev is not None and abs(ch.fader_position - prev) > 0.02:
                moving = True
            self._prev_fader_positions[ch.id] = ch.fader_position
        return moving

    def draw(self, frame: np.ndarray, channels: List[ChannelState],
             master: ChannelState) -> np.ndarray:
        for ch in channels:
            self._draw_channel(frame, ch, self._config_for(ch.id))
        self._draw_channel(frame, master, self._master_config, label="MST")
        return frame

    def _analyze_channel(self, frame: np.ndarray, cfg: dict) -> ChannelState:
        ch_id = str(cfg.get("id", "mst"))
        fader_pos = self._detect_fader_position(frame, cfg["fader_roi"])
        clipping = self._detect_clipping(frame, cfg.get("clip_roi", cfg["vu_roi"]))
        return ChannelState(
            id=ch_id,
            fader_position=fader_pos,
            is_active=fader_pos >= self._fader_active_min,
            is_clipping=clipping,
        )

    def _detect_fader_position(self, frame: np.ndarray, roi: List[int]) -> float:
        """Estima a posição do fader (0.0–1.0) pelo centróide do maior contorno claro na ROI."""
        crop = self._crop(frame, roi)
        if crop is None or crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return 0.0
        cy = M["m01"] / M["m00"]
        # cy próximo ao topo (y baixo) = fader alto = canal ativo
        return float(np.clip(1.0 - (cy / crop.shape[0]), 0.0, 1.0))

    def _detect_clipping(self, frame: np.ndarray, roi: List[int]) -> bool:
        """Detecta clipping pela proporção de pixels vermelhos (HSV) na ROI do indicador."""
        crop = self._crop(frame, roi)
        if crop is None or crop.size == 0:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lower1 = np.array([CLIPPING_HUE_LOW_1, CLIPPING_SAT_MIN, CLIPPING_VAL_MIN])
        upper1 = np.array([CLIPPING_HUE_HIGH_1, 255, 255])
        lower2 = np.array([CLIPPING_HUE_LOW_2, CLIPPING_SAT_MIN, CLIPPING_VAL_MIN])
        upper2 = np.array([CLIPPING_HUE_HIGH_2, 255, 255])
        mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
        return float(np.count_nonzero(mask)) / mask.size > self._clip_ratio_threshold

    def _draw_channel(self, frame: np.ndarray, state: ChannelState,
                      cfg: dict, label: str = "") -> None:
        roi = cfg.get("fader_roi")
        if roi is None:
            return
        x1, y1, x2, y2 = roi
        color = (0, 0, 255) if state.is_clipping else ((0, 255, 0) if state.is_active else (100, 100, 100))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        bar_h = int((y2 - y1) * state.fader_position)
        cv2.rectangle(frame, (x1, y2 - bar_h), (x2, y2), color, -1)
        name = label or f"CH{state.id}"
        suffix = " CLIP!" if state.is_clipping else ""
        cv2.putText(frame, f"{name}{suffix}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    def _config_for(self, ch_id: str) -> dict:
        for cfg in self._channels_config:
            if str(cfg.get("id")) == ch_id:
                return cfg
        return {}

    @staticmethod
    def _crop(frame: np.ndarray, roi: List[int]):
        x1, y1, x2, y2 = roi
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]
