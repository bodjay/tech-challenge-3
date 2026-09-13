from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from .person_tracker import IDetector


@dataclass
class HandResult:
    detected: bool
    positions: List[Tuple[float, float]] = field(default_factory=list)
    velocity: float = 0.0
    zone: str = "unknown"


class HandTracker(IDetector):

    def __init__(self, zones: Dict[str, List[int]], frame_shape: Tuple[int, int],
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5) -> None:
        self._mp_hands = mp.solutions.hands
        self._hands = mp.solutions.hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._drawing = mp.solutions.drawing_utils
        self._zones = zones
        self._frame_h, self._frame_w = frame_shape
        self._prev_positions: List[Tuple[float, float]] = []

    def detect(self, frame: np.ndarray) -> HandResult:
        """Detecta mãos, calcula velocidade e identifica zona do mixer."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            self._prev_positions = []
            return HandResult(detected=False)

        positions = [
            self._landmark_pos(hand.landmark, self._mp_hands.HandLandmark.INDEX_FINGER_TIP)
            for hand in results.multi_hand_landmarks
        ]
        all_points = positions + [
            self._landmark_pos(hand.landmark, self._mp_hands.HandLandmark.WRIST)
            for hand in results.multi_hand_landmarks
        ] + [
            self._landmark_pos(hand.landmark, self._mp_hands.HandLandmark.MIDDLE_FINGER_TIP)
            for hand in results.multi_hand_landmarks
        ]

        velocity = self._compute_velocity(positions)
        zone = self._identify_zone(all_points)
        self._prev_positions = positions
        return HandResult(detected=True, positions=positions, velocity=velocity, zone=zone)

    def draw(self, frame: np.ndarray, result: HandResult,
             mp_results: object = None) -> np.ndarray:
        """Desenha posição das mãos e velocidade no frame."""
        if not result.detected:
            return frame
        h, w = frame.shape[:2]
        for x_norm, y_norm in result.positions:
            cv2.circle(frame, (int(x_norm * w), int(y_norm * h)), 10, (255, 165, 0), -1)
        cv2.putText(frame, f"Vel: {result.velocity:.3f} | Zona: {result.zone}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 165, 0), 2, cv2.LINE_AA)
        return frame

    def close(self) -> None:
        self._hands.close()

    def _landmark_pos(self, landmarks, landmark_id) -> Tuple[float, float]:
        """Extrai coordenadas normalizadas (x, y) de um landmark pelo ID."""
        lm = landmarks[landmark_id]
        return (lm.x, lm.y)

    def _compute_velocity(self, current: List[Tuple[float, float]]) -> float:
        """Calcula velocidade média das mãos pela distância euclidiana entre frames."""
        if not self._prev_positions or not current:
            return 0.0
        distances = [
            np.hypot(cx - px, cy - py)
            for (cx, cy), (px, py) in zip(current, self._prev_positions)
        ]
        return float(np.mean(distances)) if distances else 0.0

    def _identify_zone(self, positions: List[Tuple[float, float]]) -> str:
        """Retorna a zona do mixer onde algum ponto de mão foi detectado."""
        for x_norm, y_norm in positions:
            px, py = int(x_norm * self._frame_w), int(y_norm * self._frame_h)
            for zone_name, roi in self._zones.items():
                x1, y1, x2, y2 = roi
                if x1 <= px <= x2 and y1 <= py <= y2:
                    return zone_name
        return "outside_mixer"
