import argparse
import os
import sys

import cv2
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.anomaly.anomaly_detector import AnomalyDetector, SystemState
from src.classifier.activity_classifier import ActivityClassifier
from src.detector.hand_tracker import HandTracker
from src.detector.mixer_detector import MixerDetector
from src.detector.person_tracker import PersonTracker
from src.detector.rhythm_analyzer import RhythmAnalyzer
from src.report.report_generator import ReportGenerator
from config.constants import Activity, AnomalySeverity, FADER_ACTIVE_MIN_POSITION


SEVERITY_COLORS = {
    AnomalySeverity.CRITICAL: (0, 0, 255),
    AnomalySeverity.HIGH:     (0, 100, 255),
    AnomalySeverity.MEDIUM:   (0, 200, 255),
    AnomalySeverity.LOW:      (255, 200, 0),
}

ACTIVITY_COLORS = {
    Activity.MIXING:        (0, 200, 80),
    Activity.EQ_ADJUSTMENT: (180, 80, 200),
    Activity.MONITORING:    (80, 160, 255),
    Activity.IDLE:          (150, 150, 150),
    Activity.BEAT_MATCHING: (0, 140, 255),
    Activity.SYNCED_MIX:    (0, 220, 180),
    Activity.TRANSITIONING: (0, 180, 255),
}


def load_config(path: str) -> dict:
    """Lê e retorna o arquivo YAML de configuração do mixer."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_components(cfg: dict, frame_shape: tuple):
    """Instancia e retorna todos os componentes do pipeline a partir da configuração."""
    h, w = frame_shape[:2]
    person_tracker = PersonTracker()
    hand_tracker = HandTracker(zones=cfg.get("zones", {}), frame_shape=(h, w))
    mixer_detector = MixerDetector(
        channels_config=cfg["channels"],
        master_config=cfg["master"],
        fader_active_min=cfg["thresholds"].get("fader_active_min_position", FADER_ACTIVE_MIN_POSITION),
        clip_ratio_threshold=cfg["thresholds"].get("clipping_red_ratio", 0.15),
        zones=cfg.get("zones", {}),
    )
    rhythm_analyzer = RhythmAnalyzer(
        window_frames=cfg["thresholds"].get("rhythm_window_frames", 90),
        sync_phase_tolerance=cfg["thresholds"].get("sync_phase_tolerance_frames", 3),
        sync_corr_min=cfg["thresholds"].get("sync_correlation_min", 0.6),
    )
    classifier = ActivityClassifier()
    anomaly_detector = AnomalyDetector.from_config(cfg.get("anomalies", {}))
    reporter = ReportGenerator()
    return (person_tracker, hand_tracker, mixer_detector,
            rhythm_analyzer, classifier, anomaly_detector, reporter)


def annotate_frame(frame, pose_result, hand_result, channels, master,
                   activity, anomalies, person_tracker, hand_tracker, mixer_detector):
    """Sobrepõe HUD de atividade, estado dos canais e alertas de anomalia no frame."""
    person_tracker.draw(frame, pose_result)
    hand_tracker.draw(frame, hand_result)
    mixer_detector.draw(frame, channels, master)

    act_color = ACTIVITY_COLORS.get(activity, (200, 200, 200))
    cv2.rectangle(frame, (0, 0), (400, 40), (20, 20, 20), -1)
    cv2.putText(frame, f"ATIVIDADE: {activity.value}", (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, act_color, 2, cv2.LINE_AA)

    for i, anomaly in enumerate(anomalies):
        color = SEVERITY_COLORS.get(anomaly.severity, (200, 200, 200))
        y = 70 + i * 28
        cv2.rectangle(frame, (0, y - 20), (500, y + 8), (20, 20, 20), -1)
        label = f"[{anomaly.severity.value}] {anomaly.type}: {anomaly.detail[:50]}"
        cv2.putText(frame, label, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, color, 1, cv2.LINE_AA)


def process_video(input_source, cfg: dict, output_dir: str, show: bool,
                  flip: bool = False) -> None:
    """Executa o pipeline completo: captura, detecção, classificação, anotação e relatório."""
    is_camera = str(input_source).isdigit()
    cap = cv2.VideoCapture(int(input_source) if is_camera else input_source)

    if not cap.isOpened():
        print(f"Erro ao abrir: {input_source}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "video_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print("Erro: nao foi possivel abrir o VideoWriter")
        return

    (person_tracker, hand_tracker, mixer_detector,
     rhythm_analyzer, classifier, anomaly_detector, reporter) = build_components(
        cfg, (height, width))

    inactivity_timer = 0.0
    frame_idx = 0
    channels_in_sync = True
    sync_phase_offset = 0.0

    iterator = tqdm(range(total_frames), desc="Processando") if not is_camera else iter(int, 1)

    for _ in iterator:
        ret, frame = cap.read()
        if not ret:
            break

        if flip:
            frame = cv2.flip(frame, -1)

        timestamp = frame_idx / fps

        pose_result = person_tracker.detect(frame)
        hand_result = hand_tracker.detect(frame)
        physical = mixer_detector.detect_physical(frame)
        channels, master = physical.channels, physical.master

        for ch in channels:
            roi = _crop(frame, _vu_roi(cfg, ch.id))
            if roi is not None:
                rhythm_analyzer.update(ch.id, roi)

        active_ids = [ch.id for ch in channels if ch.is_active]
        if len(active_ids) >= 2:
            channels_in_sync, sync_phase_offset = rhythm_analyzer.are_in_sync(
                active_ids[0], active_ids[1])

        if hand_result.detected:
            inactivity_timer = 0.0
        else:
            inactivity_timer += 1.0 / fps

        activity = classifier.classify(physical, hand_result, channels_in_sync)

        state = SystemState(
            timestamp=timestamp,
            channels=channels,
            master=master,
            hand_velocity=hand_result.velocity,
            inactivity_timer=inactivity_timer,
            channels_in_sync=channels_in_sync,
            sync_phase_offset=sync_phase_offset,
        )
        anomalies = anomaly_detector.evaluate(state)

        annotate_frame(frame, pose_result, hand_result, channels, master,
                       activity, anomalies, person_tracker, hand_tracker, mixer_detector)

        faders = {ch.id: ch.fader_position for ch in channels}
        faders["master"] = master.fader_position
        vu_snap = {ch.id: rhythm_analyzer.get_signal(ch.id) for ch in channels}
        reporter.record_frame(timestamp, faders, vu_snap, activity, anomalies)

        out.write(frame)

        if show:
            cv2.imshow("DJ Performance Analyzer", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    out.release()
    if show:
        cv2.destroyAllWindows()

    person_tracker.close()
    hand_tracker.close()

    _reencode_h264(out_path)

    print(f"\nVideo salvo em: {out_path}")
    reporter.generate(os.path.join(output_dir, "report.png"))


def _reencode_h264(video_path: str) -> None:
    """Re-encoda o vídeo de mp4v para H.264 via FFmpeg para compatibilidade com browsers."""
    import subprocess
    tmp = video_path.replace(".mp4", "_h264tmp.mp4")
    try:
        r = subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-movflags", "+faststart", "-an", tmp,
        ], capture_output=True)
        if r.returncode == 0:
            os.replace(tmp, video_path)
        elif os.path.exists(tmp):
            os.remove(tmp)
    except FileNotFoundError:
        pass


def _vu_roi(cfg: dict, ch_id: str):
    for ch in cfg.get("channels", []):
        if str(ch.get("id")) == ch_id:
            return ch.get("vu_roi")
    return None


def _crop(frame, roi):
    if roi is None:
        return None
    x1, y1, x2, y2 = roi
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def main() -> None:
    parser = argparse.ArgumentParser(description="DJ Performance Analyzer")
    parser.add_argument("--input",  required=True)
    parser.add_argument("--config", default="config/mixer_config.yaml")
    parser.add_argument("--output", default="output")
    parser.add_argument("--show",   action="store_true")
    parser.add_argument("--flip",   action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    process_video(args.input, cfg, args.output, args.show, args.flip)


if __name__ == "__main__":
    main()
