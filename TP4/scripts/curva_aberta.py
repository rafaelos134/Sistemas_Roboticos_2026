"""Curva pulso x velocidade em malha aberta (TP4, secao 4).

Le dados/malha_aberta_A50_3.csv (comando 'A 50 3' do firmware), calcula a
velocidade em regime de cada pulso (descartando 1,5 s de transitorio) como
k * delta_ticks / delta_t, grava dados/curva_aberta.csv e figuras/curva_aberta.png.
"""
import csv
import math
import os
import statistics as st

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
D_MM, N_TICKS = 43.40, 287.3
K = math.pi * D_MM / N_TICKS
NEUTRO_US = 1495
DESCARTE_S = 1.5

grupos = {}
with open(os.path.join(RAIZ, "dados", "malha_aberta_A50_3.csv")) as f:
    for r in csv.DictReader(f):
        grupos.setdefault(int(float(r["pulso_us"])), []).append(
            (float(r["t_s"]), float(r["ticks"]), float(r["v_bruta"])))

saida = []
for pulso, L in grupos.items():
    t0 = L[0][0]
    reg = [x for x in L if x[0] >= t0 + DESCARTE_S]
    dtk = reg[-1][1] - reg[0][1]
    dt = reg[-1][0] - reg[0][0]
    saida.append((pulso, K * dtk / dt, st.stdev(x[2] for x in reg)))

with open(os.path.join(RAIZ, "dados", "curva_aberta.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["pulso_us", "v_regime_mm_s", "desvio_v_bruta_mm_s"])
    for p, v, s in saida:
        w.writerow([p, f"{v:.1f}", f"{s:.1f}"])
        print(f"{p:5d}  {v:8.1f}  {s:5.1f}")

if plt is None:
    raise SystemExit("matplotlib ausente: tabela gerada, sem figura")

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot([p for p, _, _ in saida], [v for _, v, _ in saida], "o-")
ax.axvline(NEUTRO_US, color="gray", ls="--", label=f"neutro = {NEUTRO_US} us")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("Largura do pulso (us)")
ax.set_ylabel("Velocidade em regime (mm/s)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(RAIZ, "figuras", "curva_aberta.png"), dpi=200)
