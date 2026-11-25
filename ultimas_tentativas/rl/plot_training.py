"""Learning curves of the sumo runs -> docs/img/sumo_treino.png"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RUNS = [("sumo_ppo_v1", "recompensa base + shaping", "#2a78d6"),
        ("sumo_ppo_noshaping", "só recompensa base", "#e07b2a")]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for name, label, color in RUNS:
    rows = [r for r in csv.DictReader(open(HERE / "runs" / name / "log.csv")) if int(r["episodes"]) >= 20]
    x = [int(r["steps"]) / 1e6 for r in rows]
    ax[0].plot(x, [100 * float(r["win"]) for r in rows], color=color, label=label, lw=1.6)
    ax[1].plot(x, [float(r["ep_len_s"]) for r in rows], color=color, label=label, lw=1.6)
ax[0].set(xlabel="passos de simulação (milhões)", ylabel="vitórias (%)", title="Taxa de vitória (treino)")
ax[1].set(xlabel="passos de simulação (milhões)", ylabel="duração do episódio (s)", title="Duração do episódio")
for a in ax:
    a.grid(alpha=0.3)
    a.legend(frameon=False)
fig.tight_layout()
out = HERE / "docs" / "img"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "sumo_treino.png", dpi=130)
print("saved", out / "sumo_treino.png")
