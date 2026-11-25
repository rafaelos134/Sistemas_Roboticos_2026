"""Curves of phases 2 and 3 -> docs/img/sumo_fase2.png, sumo_fase3.png"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "docs" / "img"
COL = {"stopped": "#7f7f7f", "chaser": "#d62728", "careful": "#ff7f0e", "evader": "#2ca02c",
       "random": "#9467bd", "self": "#1f77b4"}
LBL = {"stopped": "parado", "chaser": "perseguidor", "careful": "perseguidor cuidadoso", "evader": "esquiva",
       "random": "aleatório", "self": "cópias da própria política"}


def smooth(y, k=15):
    out, acc = [], []
    for v in y:
        acc.append(v)
        acc = acc[-k:]
        out.append(sum(acc) / len(acc))
    return out


for run, fname, title in (("sumo_ppo_v2", "sumo_fase2.png", "Fase 2: adversários com script"),
                          ("sumo_ppo_v3", "sumo_fase3.png", "Fase 3: self-play com cópias congeladas")):
    rows = list(csv.DictReader(open(HERE / "runs" / run / "log.csv")))
    x = [int(r["steps"]) / 1e6 for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for g in COL:
        if "win_" + g in rows[0]:
            ax[0].plot(x, smooth([100 * float(r["win_" + g]) for r in rows]), color=COL[g], label=LBL[g], lw=1.5)
    for key, lbl, c in (("win", "vitória", "#1f77b4"), ("lose", "derrota", "#d62728"), ("draw", "empate", "#7f7f7f")):
        ax[1].plot(x, smooth([100 * float(r[key]) for r in rows]), color=c, label=lbl, lw=1.5)
    ax[0].set(xlabel="passos de simulação (milhões)", ylabel="vitórias (%)", title="Vitória por adversário", ylim=(-3, 103))
    ax[1].set(xlabel="passos de simulação (milhões)", ylabel="% dos episódios", title="Resultado geral", ylim=(-3, 103))
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=130)
    print("saved", OUT / fname)
