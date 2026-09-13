"""
Ferramenta de calibração das ROIs do mixer.

Uso:
    python calibrate.py --input video.mp4
    python calibrate.py --input video.mp4 --frame 30

Controles (mostrados na janela):
    Clique e arraste  — seleciona a ROI do elemento atual
    ENTER             — confirma a seleção e vai para o próximo
    C                 — cancela / reseleciona
    A / D             — frame anterior / próximo
    B / F             — recua / avança 10 frames
    Q                 — sai e salva o YAML gerado
"""

import argparse
import os
import sys

import cv2
import numpy as np

ELEMENTS = [
    ("mixer_roi",       "Area TOTAL do mixer"),
    ("zone_eq_knobs",   "Zona dos knobs de EQ"),
    ("zone_faders",     "Zona dos faders de canal"),
    ("zone_crossfader", "Zona do crossfader"),
    ("zone_headphones", "Zona do fone de ouvido"),
    ("ch1_fader",       "Fader do CANAL 1"),
    ("ch1_vu",          "VU meter do CANAL 1"),
    ("ch1_clip",        "Clip indicator CANAL 1 (led vermelho)"),
    ("ch1_jog",         "Jog wheel do CANAL 1  [ENTER para pular]"),
    ("ch2_fader",       "Fader do CANAL 2"),
    ("ch2_vu",          "VU meter do CANAL 2"),
    ("ch2_clip",        "Clip indicator CANAL 2"),
    ("ch2_jog",         "Jog wheel do CANAL 2  [ENTER para pular]"),
    ("master_fader",    "Fader do MASTER"),
    ("master_vu",       "VU meter do MASTER"),
    ("master_clip",     "Clip indicator MASTER"),
]

WIN = "Calibrador de ROI"
COLORS = [(0, 255, 0), (0, 165, 255), (255, 200, 0), (200, 0, 255),
          (0, 200, 255), (255, 100, 100), (100, 255, 100)]


def get_frame(cap, index, is_camera=False):
    if is_camera:
        ok, frame = cap.read()
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Nao foi possivel ler o frame")
    return frame


def draw_hud(frame, name, description, collected, current_roi=None):
    h, w = frame.shape[:2]

    # Barra superior semitransparente
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 70), (15, 15, 15), -1)
    cv2.addWeighted(bar, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, f"  [{name}]  {description}",
                (4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame,
                "  Clique e arraste para selecionar  |  ENTER=confirmar  |  C=cancelar  |  A/D=frame  |  Q=sair",
                (4, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    # ROIs já coletadas
    for i, (k, roi) in enumerate(collected.items()):
        x1, y1, x2, y2 = roi
        c = COLORS[i % len(COLORS)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 1)
        cv2.putText(frame, k, (x1 + 2, max(y1 + 13, 75)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, c, 1, cv2.LINE_AA)

    # ROI em andamento
    if current_roi and current_roi[2] > 0:
        x, y, x2, y2 = current_roi
        cv2.rectangle(frame, (x, y), (x2, y2), (255, 255, 255), 2)


# ── Estado do mouse ────────────────────────────────────────────────────────────

class MouseState:
    def __init__(self):
        self.drawing = False
        self.start = (0, 0)
        self.end = (0, 0)
        self.confirmed = None  # roi confirmada pelo usuário

    def roi_rect(self):
        x1 = min(self.start[0], self.end[0])
        y1 = min(self.start[1], self.end[1])
        x2 = max(self.start[0], self.end[0])
        y2 = max(self.start[1], self.end[1])
        return (x1, y1, x2, y2)

    def reset(self):
        self.drawing = False
        self.start = (0, 0)
        self.end = (0, 0)
        self.confirmed = None


def make_mouse_cb(ms: MouseState):
    def cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            ms.drawing = True
            ms.start = (x, y)
            ms.end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and ms.drawing:
            ms.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            ms.drawing = False
            ms.end = (x, y)
    return cb


# ── Loop principal ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Calibrador de ROIs do mixer")
    parser.add_argument("--input",  required=True, help="Caminho do video")
    parser.add_argument("--frame",  type=int, default=0, help="Frame inicial")
    parser.add_argument("--out",    default="config/mixer_config.yaml",
                        help="Arquivo YAML de saida")
    parser.add_argument("--flip",   action="store_true",
                        help="Vira o frame 180 graus (camera de ponta cabeca)")
    args = parser.parse_args()

    source = int(args.input) if args.input.isdigit() else args.input
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Erro: nao foi possivel abrir '{args.input}'"); sys.exit(1)

    is_camera = args.input.isdigit()
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_camera else 1
    frame_idx = max(0, min(args.frame, total - 1))
    frame     = get_frame(cap, frame_idx, is_camera)
    if args.flip:
        frame = cv2.flip(frame, -1)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, min(1280, frame.shape[1]), min(800, frame.shape[0]))

    ms = MouseState()
    cv2.setMouseCallback(WIN, make_mouse_cb(ms))

    collected: dict = {}

    print(f"\n=== CALIBRADOR DE ROI — DJ Performance Analyzer ===")
    print(f"Video: {args.input}  ({total} frames)")
    print("Clique e arraste diretamente na janela para marcar cada elemento.\n")

    for name, description in ELEMENTS:
        ms.reset()
        print(f"[{name}] {description}")

        while True:
            base = frame.copy()
            current = ms.roi_rect() if (ms.drawing or ms.end != (0, 0)) else None
            draw_hud(base, name, description, collected, current)
            cv2.imshow(WIN, base)

            key = cv2.waitKey(30) & 0xFF

            # Sair
            if key == ord("q"):
                cap.release(); cv2.destroyAllWindows()
                _finish(collected, args.out); return

            # Câmera: sempre lê frame novo (feed ao vivo)
            if is_camera:
                frame = get_frame(cap, frame_idx, is_camera=True)
                if args.flip:
                    frame = cv2.flip(frame, -1)

            # Navegação de frames (só para vídeo)
            if not is_camera and key in (ord("d"), 83):
                frame_idx = min(frame_idx + 1, total - 1)
                frame = get_frame(cap, frame_idx)
                if args.flip: frame = cv2.flip(frame, -1)
                ms.reset(); print(f"  frame {frame_idx}")
            elif not is_camera and key in (ord("a"), 81):
                frame_idx = max(frame_idx - 1, 0)
                frame = get_frame(cap, frame_idx)
                if args.flip: frame = cv2.flip(frame, -1)
                ms.reset(); print(f"  frame {frame_idx}")
            elif not is_camera and key == ord("f"):
                frame_idx = min(frame_idx + 10, total - 1)
                frame = get_frame(cap, frame_idx)
                if args.flip: frame = cv2.flip(frame, -1)
                ms.reset(); print(f"  frame {frame_idx}")
            elif not is_camera and key == ord("b"):
                frame_idx = max(frame_idx - 10, 0)
                frame = get_frame(cap, frame_idx)
                if args.flip: frame = cv2.flip(frame, -1)
                ms.reset(); print(f"  frame {frame_idx}")

            # Cancelar seleção atual
            if key == ord("c"):
                ms.reset()

            # ENTER — confirma roi atual ou pula
            elif key == 13:
                x1, y1, x2, y2 = ms.roi_rect()
                w_ = x2 - x1; h_ = y2 - y1
                if w_ > 2 and h_ > 2:
                    collected[name] = [x1, y1, x2, y2]
                    print(f"  {name}: {[x1, y1, x2, y2]}")
                else:
                    print(f"  {name}: pulado")
                break

    cap.release()
    cv2.destroyAllWindows()
    _finish(collected, args.out)


def _finish(rois: dict, out_path: str) -> None:
    import yaml as _yaml

    # Carrega config existente para preservar thresholds e anomalies
    existing: dict = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing = _yaml.safe_load(f) or {}
        except Exception:
            pass

    DEFAULT_THRESHOLDS = {
        "fader_active_min_position": 0.15,
        "clipping_red_ratio": 0.15,
        "inactivity_timeout_seconds": 30.0,
        "sync_phase_tolerance_frames": 3,
        "sync_correlation_min": 0.6,
        "hand_velocity_threshold": 0.04,
        "rhythm_window_frames": 90,
    }
    DEFAULT_ANOMALIES = {
        "MASTER_OFF":          {"enabled": False},
        "CHANNEL_CLIPPING":    {"enabled": True},
        "ALL_CHANNELS_MUTED":  {"enabled": True},
        "BEAT_MISMATCH":       {"enabled": False},
        "ABRUPT_MOVEMENT":     {"enabled": False},
        "EXTENDED_INACTIVITY": {"enabled": True},
    }

    thresholds = existing.get("thresholds", DEFAULT_THRESHOLDS)
    anomalies  = existing.get("anomalies",  DEFAULT_ANOMALIES)

    def r(k):
        return rois.get(k, [0, 0, 0, 0])

    def ch_block(n):
        f = r(f"ch{n}_fader")
        if f == [0, 0, 0, 0]:
            return ""
        return (f'  - id: "{n}"\n'
                f'    fader_roi: {f}\n'
                f'    vu_roi:    {r(f"ch{n}_vu")}\n'
                f'    clip_roi:  {r(f"ch{n}_clip")}\n'
                f'    jog_roi:   {r(f"ch{n}_jog")}\n')

    channels_yaml = "".join(filter(None, [ch_block(n) for n in range(1, 3)]))

    def fmt_thresholds(t: dict) -> str:
        return "\n".join(f"  {k}: {v}" for k, v in t.items())

    def fmt_anomalies(a: dict) -> str:
        return "\n".join(
            f"  {k}:{{\"enabled\": {str(v.get('enabled', True)).lower()}}}"
            if isinstance(v, dict) else f"  {k}: {{enabled: true}}"
            for k, v in a.items()
        )

    # Formata anomalies de forma legível
    anom_lines = []
    for k, v in anomalies.items():
        enabled = v.get("enabled", True) if isinstance(v, dict) else True
        anom_lines.append(f"  {k}:          {{enabled: {str(enabled).lower()}}}")
    anom_yaml = "\n".join(anom_lines)

    yaml_text = f"""\
# Gerado por calibrate.py

mixer_roi: {r("mixer_roi")}

zones:
  eq_knobs:   {r("zone_eq_knobs")}
  faders:     {r("zone_faders")}
  crossfader: {r("zone_crossfader")}
  headphones: {r("zone_headphones")}

channels:
{channels_yaml}
master:
  fader_roi: {r("master_fader")}
  vu_roi:    {r("master_vu")}
  clip_roi:  {r("master_clip")}

thresholds:
{fmt_thresholds(thresholds)}

anomalies:
{anom_yaml}
"""

    print("\n" + "=" * 60)
    print("YAML GERADO:")
    print("=" * 60)
    print(yaml_text)

    try:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(yaml_text)
        print(f"Salvo em: {out_path}")
    except Exception as e:
        print(f"Nao foi possivel salvar: {e}")
        print("Copie o YAML acima manualmente.")


if __name__ == "__main__":
    main()
