"""Grafico da comparacao de filtros no degrau (TP4, secao 5).

Le dados/filtros_degrau_o1700.csv e gera figuras/filtros_degrau.png:
  esquerda: degrau (0 a 1,5 s) com a velocidade bruta e as duas filtradas;
  direita:  regime (5 a 8 s), onde se ve o ruido que cada filtro deixa passar.
Uso: uv run python scripts/grafico_filtros.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
t, bruta, mm, lp = [], [], [], []
with open(os.path.join(RAIZ, "dados", "filtros_degrau_o1700.csv")) as f:
    for r in csv.DictReader(f):
        t.append(float(r["t_s"]))
        bruta.append(float(r["v_bruta"]))
        mm.append(float(r["v_mm"]))
        lp.append(float(r["v_lp"]))

fig, eixos = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, (t0, t1), titulo in zip(eixos, [(0, 1.5), (5, 8)], ["Degrau (0 a 1,5 s)", "Regime (5 a 8 s)"]):
    idx = [i for i, x in enumerate(t) if t0 <= x <= t1]
    ts = [t[i] for i in idx]
    ax.plot(ts, [bruta[i] for i in idx], color="0.65", lw=1, marker=".", ms=4, label="bruta")
    ax.plot(ts, [mm[i] for i in idx], color="tab:blue", lw=1.8, label="média móvel (n = 5)")
    ax.plot(ts, [lp[i] for i in idx], color="tab:orange", lw=1.8, ls="--", label="passa-baixa (α = 0,3)")
    ax.set_title(titulo)
    ax.set_xlabel("Tempo (s)")
    ax.grid(alpha=0.3)
eixos[0].set_ylabel("Velocidade (mm/s)")
eixos[0].legend(loc="lower right")
fig.suptitle("Degrau em malha aberta, pulso de 1700 µs a partir do repouso")
fig.tight_layout()
saida = os.path.join(RAIZ, "figuras", "filtros_degrau.png")
fig.savefig(saida, dpi=200)
print(saida)
