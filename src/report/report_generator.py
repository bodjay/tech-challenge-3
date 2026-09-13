from __future__ import annotations

import csv
import os
from abc import ABC, abstractmethod
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from config.constants import Activity, AnomalySeverity
from src.anomaly.anomaly_detector import Anomaly


SEVERITY_COLORS = {
    AnomalySeverity.CRITICAL: "#e74c3c",
    AnomalySeverity.HIGH:     "#e67e22",
    AnomalySeverity.MEDIUM:   "#f1c40f",
    AnomalySeverity.LOW:      "#3498db",
}

ACTIVITY_COLORS = {
    Activity.MIXING:        "#2ecc71",
    Activity.EQ_ADJUSTMENT: "#9b59b6",
    Activity.MONITORING:    "#3498db",
    Activity.IDLE:          "#95a5a6",
    Activity.BEAT_MATCHING: "#e67e22",
    Activity.SYNCED_MIX:    "#1abc9c",
    Activity.TRANSITIONING: "#f39c12",
}


class IReportGenerator(ABC):
    @abstractmethod
    def generate(self, output_path: str) -> None:
        pass


class ReportGenerator(IReportGenerator):
    """Gera relatório visual com gráficos de faders, VU meters, atividades e anomalias."""

    def __init__(self) -> None:
        self.fader_history: Dict[str, List[float]] = {}
        self.vu_signals: Dict[str, List[float]] = {}
        self.activity_history: List[str] = []
        self.anomalies: List[Anomaly] = []
        self.timestamps: List[float] = []

    def record_frame(self, timestamp: float, faders: Dict[str, float],
                     vu_signals: Dict[str, List[float]],
                     activity: Activity, anomalies: List[Anomaly]) -> None:
        self.timestamps.append(timestamp)
        for ch_id, pos in faders.items():
            self.fader_history.setdefault(ch_id, []).append(pos)
        for ch_id, sig in vu_signals.items():
            if sig:
                self.vu_signals.setdefault(ch_id, []).append(sig[-1])
        self.activity_history.append(activity.value)
        self.anomalies.extend(anomalies)

    def generate(self, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig, axes = plt.subplots(4, 1, figsize=(14, 18))
        fig.suptitle("DJ Performance Analyzer — Relatório", fontsize=16, fontweight="bold")

        self._plot_faders(axes[0])
        self._plot_vu_meters(axes[1])
        self._plot_activity_timeline(axes[2])
        self._plot_anomaly_timeline(axes[3])

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Relatorio salvo em: {output_path}")

        csv_path = os.path.splitext(output_path)[0] + "_anomalias.csv"
        self._export_anomalies_csv(csv_path)
        self._print_summary()

    def _export_anomalies_csv(self, csv_path: str) -> None:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_s", "tipo", "severidade", "detalhe"])
            for a in self.anomalies:
                writer.writerow([
                    f"{a.timestamp:.3f}",
                    a.type,
                    a.severity.value,
                    a.detail,
                ])
        print(f"CSV de anomalias salvo em: {csv_path}")

    # ------------------------------------------------------------------

    def _plot_faders(self, ax: plt.Axes) -> None:
        t = self.timestamps
        for ch_id, positions in self.fader_history.items():
            ax.plot(t[:len(positions)], positions, label=f"Canal {ch_id}", linewidth=1.2)
        ax.set_title("Posicao dos Faders ao Longo do Tempo")
        ax.set_ylabel("Posicao (0=mudo, 1=cheio)")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.15, color="gray", linestyle="--", linewidth=0.8, label="Min ativo")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    def _plot_vu_meters(self, ax: plt.Axes) -> None:
        t = self.timestamps
        for ch_id, signal in self.vu_signals.items():
            ax.plot(t[:len(signal)], signal, label=f"VU {ch_id}", linewidth=0.8, alpha=0.8)
        ax.set_title("Sinal Visual dos VU Meters (luminosidade)")
        ax.set_ylabel("Brilho medio (0-255)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    def _plot_activity_timeline(self, ax: plt.Axes) -> None:
        if not self.activity_history:
            return
        t = np.array(self.timestamps[:len(self.activity_history)])
        activities = list(Activity)
        act_to_y = {a.value: i for i, a in enumerate(activities)}

        prev_t, prev_a = t[0], self.activity_history[0]
        for i in range(1, len(self.activity_history)):
            if self.activity_history[i] != prev_a or i == len(self.activity_history) - 1:
                color = ACTIVITY_COLORS.get(Activity(prev_a), "#cccccc")
                ax.barh(act_to_y.get(prev_a, 0), t[i] - prev_t,
                        left=prev_t, height=0.6, color=color, alpha=0.85)
                prev_t, prev_a = t[i], self.activity_history[i]

        ax.set_yticks(range(len(activities)))
        ax.set_yticklabels([a.value for a in activities], fontsize=8)
        ax.set_title("Timeline de Atividades")
        ax.grid(axis="x", alpha=0.3)

    def _plot_anomaly_timeline(self, ax: plt.Axes) -> None:
        if not self.anomalies:
            ax.text(0.5, 0.5, "Nenhuma anomalia detectada", ha="center",
                    va="center", transform=ax.transAxes, fontsize=12, color="green")
            ax.set_title("Timeline de Anomalias")
            return

        anomaly_types = list({a.type for a in self.anomalies})
        type_to_y = {t: i for i, t in enumerate(anomaly_types)}

        for anomaly in self.anomalies:
            color = SEVERITY_COLORS.get(anomaly.severity, "#999999")
            y = type_to_y[anomaly.type]
            ax.scatter(anomaly.timestamp, y, color=color, s=80, zorder=5)

        ax.set_yticks(range(len(anomaly_types)))
        ax.set_yticklabels(anomaly_types, fontsize=8)
        ax.set_title("Timeline de Anomalias")
        ax.set_xlabel("Tempo (s)")
        ax.grid(alpha=0.3)

        patches = [
            mpatches.Patch(color=c, label=s.value)
            for s, c in SEVERITY_COLORS.items()
        ]
        ax.legend(handles=patches, loc="upper right", fontsize=8)

    def _print_summary(self) -> None:
        total = len(self.timestamps)
        print("\n=== RESUMO DA PERFORMANCE ===")
        print(f"Frames analisados : {total}")
        if self.activity_history:
            from collections import Counter
            counts = Counter(self.activity_history)
            print("\nDistribuicao de Atividades:")
            for act, n in counts.most_common():
                pct = 100 * n / total
                print(f"  {act:<20} {pct:5.1f}%")

        print(f"\nAnomalias detectadas: {len(self.anomalies)}")
        sev_count: dict[str, int] = {}
        for a in self.anomalies:
            sev_count[a.severity.value] = sev_count.get(a.severity.value, 0) + 1
        for sev, n in sev_count.items():
            print(f"  [{sev}] {n}")
        print("==============================\n")
