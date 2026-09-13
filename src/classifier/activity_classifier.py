from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import List

from config.constants import (
    ACTIVITY_SMOOTHING_FRAMES,
    Activity,
    BEAT_MATCHING_VELOCITY_MIN,
    HAND_VELOCITY_THRESHOLD,
    MIXING_VELOCITY_MIN,
)
from src.detector.hand_tracker import HandResult
from src.detector.mixer_detector import ChannelState, MixerPhysicalState


class IActivityClassifier(ABC):
    @abstractmethod
    def classify(self, physical: MixerPhysicalState,
                 hand: HandResult, in_sync: bool) -> Activity:
        pass


class ActivityClassifier(IActivityClassifier):

    def __init__(self, smoothing_frames: int = ACTIVITY_SMOOTHING_FRAMES) -> None:
        self._history: deque[Activity] = deque(maxlen=smoothing_frames)

    def classify(self, physical: MixerPhysicalState,
                 hand: HandResult, in_sync: bool) -> Activity:
        """Classifica a atividade do DJ e aplica suavização por votação majoritária."""
        raw = self._raw_classify(physical, hand, in_sync)
        self._history.append(raw)
        return self._majority_vote()

    def _raw_classify(self, physical: MixerPhysicalState,
                      hand: HandResult, in_sync: bool) -> Activity:
        """Determina a atividade com base no estado físico detectado do mixer."""
        active_channels = [ch for ch in physical.channels if ch.is_active]

        if physical.eq_active:
            return Activity.EQ_ADJUSTMENT
        if physical.crossfader_moving and len(active_channels) >= 2:
            return Activity.TRANSITIONING
        if len(active_channels) >= 2:
            return Activity.SYNCED_MIX if in_sync else Activity.TRANSITIONING
        if physical.jog_active:
            return Activity.BEAT_MATCHING
        if physical.faders_moving:
            return Activity.MIXING
        if hand.detected and hand.zone == "headphones":
            return Activity.MONITORING
        return Activity.IDLE

    def _majority_vote(self) -> Activity:
        """Retorna a atividade mais frequente nos últimos N frames."""
        if not self._history:
            return Activity.IDLE
        counts: dict[Activity, int] = {}
        for activity in self._history:
            counts[activity] = counts.get(activity, 0) + 1
        return max(counts, key=counts.__getitem__)
