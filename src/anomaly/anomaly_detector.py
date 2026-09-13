from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from config.constants import (
    AnomalySeverity,
    FADER_ACTIVE_MIN_POSITION,
    HAND_VELOCITY_THRESHOLD,
    INACTIVITY_TIMEOUT_SECONDS,
)
from src.detector.mixer_detector import ChannelState


@dataclass
class Anomaly:
    type: str
    severity: AnomalySeverity
    timestamp: float
    detail: str = ""


@dataclass
class SystemState:
    timestamp: float
    channels: List[ChannelState]
    master: ChannelState
    hand_velocity: float
    inactivity_timer: float
    channels_in_sync: bool
    sync_phase_offset: float
    active_channel_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.active_channel_ids = [ch.id for ch in self.channels if ch.is_active]


class AnomalyRule(ABC):
    @abstractmethod
    def evaluate(self, state: SystemState) -> Optional[Anomaly]:
        pass


class MasterOffRule(AnomalyRule):
    def evaluate(self, state: SystemState) -> Optional[Anomaly]:
        if state.master.fader_position < FADER_ACTIVE_MIN_POSITION:
            return Anomaly(
                type="MASTER_OFF",
                severity=AnomalySeverity.CRITICAL,
                timestamp=state.timestamp,
                detail="Canal master desligado durante a performance.",
            )
        return None


class ChannelClippingRule(AnomalyRule):
    def evaluate(self, state: SystemState) -> Optional[Anomaly]:
        clipping = [ch for ch in state.channels if ch.is_clipping]
        if not clipping:
            return None
        ids = ", ".join(ch.id for ch in clipping)
        return Anomaly(
            type="CHANNEL_CLIPPING",
            severity=AnomalySeverity.CRITICAL,
            timestamp=state.timestamp,
            detail=f"Ganho excessivo no(s) canal(is): {ids}.",
        )


class AllChannelsMutedRule(AnomalyRule):
    def evaluate(self, state: SystemState) -> Optional[Anomaly]:
        if all(not ch.is_active for ch in state.channels) and \
                state.master.fader_position >= FADER_ACTIVE_MIN_POSITION:
            return Anomaly(
                type="ALL_CHANNELS_MUTED",
                severity=AnomalySeverity.HIGH,
                timestamp=state.timestamp,
                detail="Todos os canais silenciados simultaneamente.",
            )
        return None


class BeatMismatchRule(AnomalyRule):
    def evaluate(self, state: SystemState) -> Optional[Anomaly]:
        if len(state.active_channel_ids) >= 2 and not state.channels_in_sync:
            offset = abs(state.sync_phase_offset)
            return Anomaly(
                type="BEAT_MISMATCH",
                severity=AnomalySeverity.HIGH,
                timestamp=state.timestamp,
                detail=f"Canais {state.active_channel_ids[:2]} fora de fase ({offset:.0f} frames).",
            )
        return None


class AbruptMovementRule(AnomalyRule):
    def __init__(self, velocity_threshold: float = HAND_VELOCITY_THRESHOLD) -> None:
        self._threshold = velocity_threshold

    def evaluate(self, state: SystemState) -> Optional[Anomaly]:
        if state.hand_velocity > self._threshold:
            return Anomaly(
                type="ABRUPT_MOVEMENT",
                severity=AnomalySeverity.MEDIUM,
                timestamp=state.timestamp,
                detail=f"Movimento brusco das maos (vel={state.hand_velocity:.3f}).",
            )
        return None


class ExtendedInactivityRule(AnomalyRule):
    def __init__(self, timeout: float = INACTIVITY_TIMEOUT_SECONDS) -> None:
        self._timeout = timeout

    def evaluate(self, state: SystemState) -> Optional[Anomaly]:
        if state.inactivity_timer >= self._timeout:
            return Anomaly(
                type="EXTENDED_INACTIVITY",
                severity=AnomalySeverity.LOW,
                timestamp=state.timestamp,
                detail=f"DJ sem interacao por {state.inactivity_timer:.0f}s.",
            )
        return None


class AnomalyDetector:

    def __init__(self, rules: List[AnomalyRule]) -> None:
        self._rules = rules

    def evaluate(self, state: SystemState) -> List[Anomaly]:
        """Avalia todas as regras ativas contra o estado atual e retorna as anomalias disparadas."""
        return [r for rule in self._rules for r in [rule.evaluate(state)] if r is not None]

    @classmethod
    def default(cls) -> "AnomalyDetector":
        return cls.from_config({})

    @classmethod
    def from_config(cls, anomaly_cfg: dict) -> "AnomalyDetector":
        """Cria o detector incluindo apenas as regras habilitadas no YAML de configuração."""
        def enabled(name: str) -> bool:
            return anomaly_cfg.get(name, {}).get("enabled", True)

        rules: List[AnomalyRule] = []
        if enabled("MASTER_OFF"):          rules.append(MasterOffRule())
        if enabled("CHANNEL_CLIPPING"):    rules.append(ChannelClippingRule())
        if enabled("ALL_CHANNELS_MUTED"):  rules.append(AllChannelsMutedRule())
        if enabled("BEAT_MISMATCH"):       rules.append(BeatMismatchRule())
        if enabled("ABRUPT_MOVEMENT"):     rules.append(AbruptMovementRule())
        if enabled("EXTENDED_INACTIVITY"): rules.append(ExtendedInactivityRule())
        return cls(rules=rules)
