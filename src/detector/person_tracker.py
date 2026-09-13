from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np


class IDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> object:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


@dataclass
class PoseResult:
    detected: bool
    landmarks: list = field(default_factory=list)
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0)


class PersonTracker(IDetector):

    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5) -> None:
        self._mp_pose = mp.solutions.pose
        self._pose = mp.solutions.pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._drawing = mp.solutions.drawing_utils

    def detect(self, frame: np.ndarray) -> PoseResult:
        """Detecta pose do DJ e retorna landmarks visíveis com bounding box."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)

        if not results.pose_landmarks:
            return PoseResult(detected=False)

        h, w = frame.shape[:2]
        lms = results.pose_landmarks.landmark
        xs = [lm.x * w for lm in lms if lm.visibility > 0.3]
        ys = [lm.y * h for lm in lms if lm.visibility > 0.3]

        if not xs:
            return PoseResult(detected=False)

        bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        return PoseResult(detected=True, landmarks=list(lms), bounding_box=bbox)

    def draw(self, frame: np.ndarray, result: PoseResult) -> np.ndarray:
        """Desenha skeleton MediaPipe e bounding box no frame."""
        if not result.detected:
            return frame
        self._drawing.draw_landmarks(
            frame, _LandmarkWrapper(result.landmarks), self._mp_pose.POSE_CONNECTIONS
        )
        x1, y1, x2, y2 = result.bounding_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "DJ", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)
        return frame

    def close(self) -> None:
        self._pose.close()


class _LandmarkWrapper:
    """Compatibilidade com mp.solutions.drawing_utils que exige .landmark."""

    def __init__(self, landmarks: list) -> None:
        self.landmark = landmarks
