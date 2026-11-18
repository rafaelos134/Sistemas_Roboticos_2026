"""Graficos dos ensaios do TP4 (secoes 7 e 8) a partir dos CSVs em dados/.

Gera em figuras/:
  curva_aberta.png          pulso x velocidade em malha aberta (secao 5)
  pid_sintonia.png          degraus T01 a T05 da sintonia (secao 7)
  ensaio1_patamares.png     cinco patamares, 100 a 160 mm/s
  ensaio1_saturado.png      corrida com 190 e 220 mm/s (saturacao em +250 us)
  ensaio2_inversao.png      inversao de sentido
  ensaio3_distancia.png     distancia s(t) com o valor esperado de tres voltas
Uso: uv run python scripts/graficos_ensaios.py
"""
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS = os.path.join(RAIZ, "dados")
FIG = os.path.join(RAIZ, "figuras")
os.makedirs(FIG, exist_ok=True)

D_MM, NEUTRO, LIM = 43.40, 1495, 250
AZUL, LARANJA, CINZA, TINTA = "#2a78d6", "#eb6834", "0.72", "#222222"

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
})


def ler(nome):
    with open(os.path.join(DADOS, nome + ".csv")) as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def col(d, k):
    return [x[k] for x in d]


def salva(fig, nome):
    caminho = os.path.join(FIG, nome + ".png")
    fig.savefig(caminho, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(caminho)


def painel_velocidade(ax, d, bruta=True):
    t = col(d, "t_s")
    if bruta:
        ax.plot(t, col(d, "v_bruta"), color=CINZA, lw=0.8, label="velocidade bruta", zorder=1)
    ax.plot(t, col(d, "v_filtrada"), color=AZUL, lw=1.8, label="velocidade filtrada", zorder=3)
    ax.step(t, col(d, "setpoint"), where="post", color=TINTA, lw=1.2, ls="--", label="setpoint", zorder=2)
    ax.set_ylabel("Velocidade (mm/s)")


def painel_pulso(ax, d):
    t = col(d, "t_s")
    ax.plot(t, col(d, "pulso_us"), color=LARANJA, lw=1.2)
    ax.axhline(NEUTRO, color="0.4", lw=0.8)
    for lim in (NEUTRO - LIM, NEUTRO + LIM):
        ax.axhline(lim, color="0.4", lw=0.8, ls=":")
    tr = ax.get_yaxis_transform()  # x em fracao do eixo, y em dados
    ax.text(1.01, NEUTRO + LIM, "+250 µs (1745)", transform=tr, va="center", fontsize=8, color="0.3")
    ax.text(1.01, NEUTRO, "neutro (1495)", transform=tr, va="center", fontsize=8, color="0.3")
    ax.text(1.01, NEUTRO - LIM, "−250 µs (1245)", transform=tr, va="center", fontsize=8, color="0.3")
    ax.set_ylabel("Pulso (µs)")
    ax.set_xlabel("Tempo (s)")


def ensaio_dois_paineis(nome_csv, nome_fig, titulo):
    d = ler(nome_csv)
    fig, (a, b) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1.6]})
    painel_velocidade(a, d)
    a.legend(loc="upper left", ncol=3, fontsize=8)
    a.set_title(titulo)
    painel_pulso(b, d)
    fig.tight_layout()
    salva(fig, nome_fig)


# --- curva aberta ---
with open(os.path.join(DADOS, "curva_aberta.csv")) as f:
    ca = [(float(r["pulso_us"]), float(r["v_regime_mm_s"])) for r in csv.DictReader(f)]
fig, ax = plt.subplots(figsize=(6.5, 4))
# a linha nao atravessa a banda morta: dois trechos e o ponto de 1500 us a parte
neg = [(p, v) for p, v in ca if p < 1500]
pos = [(p, v) for p, v in ca if p > 1500]
for seg in (neg, pos):
    ax.plot([p for p, _ in seg], [v for _, v in seg], "o-", color=AZUL, ms=4, lw=1.6)
ax.plot([1500], [0], "o", mfc="white", mec=AZUL, ms=5)
ax.axvline(NEUTRO, color="0.4", ls="--", lw=1, label=f"neutro = {NEUTRO} µs")
ax.axvspan(1465, 1525, color="0.85", alpha=0.7, label="banda morta (≈ 60 µs)")
ax.axvline(NEUTRO + LIM, color=LARANJA, ls=":", lw=1.2, label="limite da correção (±250 µs)")
ax.axvline(NEUTRO - LIM, color=LARANJA, ls=":", lw=1.2)
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("Largura do pulso (µs)")
ax.set_ylabel("Velocidade em regime (mm/s)")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
salva(fig, "curva_aberta")

# --- sintonia (T01 a T05) ---
tent = [("pid_t01", "T01: kp = 1, ki = 0"), ("pid_t02", "T02: kp = 3, ki = 0"),
        ("pid_t03", "T03: kp = 6, ki = 0"), ("pid_t04", "T04: kp = 3, ki = 3"),
        ("pid_t05", "T05: kp = 3, ki = 8")]
fig, eixos = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True, sharey=True)
for ax, (nome, tit) in zip(eixos.flat, tent):
    d = ler(nome)
    t = col(d, "t_s")
    ax.plot(t, col(d, "v_bruta"), color=CINZA, lw=0.7)
    ax.plot(t, col(d, "v_filtrada"), color=AZUL, lw=1.6)
    ax.axhline(150, color=TINTA, ls="--", lw=1)
    ax.set_title(tit)
for ax in eixos[1]:
    ax.set_xlabel("Tempo (s)")
for ax in eixos[:, 0]:
    ax.set_ylabel("Velocidade (mm/s)")
leg = eixos.flat[5]
leg.axis("off")
leg.plot([], [], color=CINZA, lw=1, label="velocidade bruta")
leg.plot([], [], color=AZUL, lw=1.8, label="velocidade filtrada")
leg.plot([], [], color=TINTA, ls="--", lw=1.2, label="setpoint (150 mm/s)")
leg.legend(loc="center", fontsize=9)
fig.tight_layout()
salva(fig, "pid_sintonia")

# --- ensaios ---
ensaio_dois_paineis("ensaio1_patamares", "ensaio1_patamares",
                    "Ensaio 1: cinco patamares (kp = 3, ki = 8, kd = 0)")
ensaio_dois_paineis("ensaio1_patamares_saturado", "ensaio1_saturado",
                    "Ensaio 1, primeira corrida: 190 e 220 mm/s saturam em +250 µs")
ensaio_dois_paineis("ensaio2_inversao", "ensaio2_inversao",
                    "Ensaio 2: inversão de sentido (+120 e −120 mm/s)")

# --- distancia ---
d = ler("ensaio3_distancia_v2")
t, s = col(d, "t_s"), col(d, "s_mm")
alvo = 3 * math.pi * D_MM
fig, (a, b) = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1.4]})
a.plot(t, s, color=AZUL, lw=1.8, label="distância s")
a.axhline(alvo, color=TINTA, ls="--", lw=1.2, label=f"3 · π · D = {alvo:.1f} mm")
a.annotate(f"s final = {s[-1]:.1f} mm", xy=(t[-1], s[-1]), xytext=(-110, -42),
           textcoords="offset points", fontsize=9,
           arrowprops=dict(arrowstyle="-", color="0.4"))
a.set_ylabel("Distância (mm)")
a.set_title("Ensaio 3: distância percorrida em três voltas a 120 mm/s")
a.set_ylim(-15, 480)
a.legend(loc="center left", fontsize=8, bbox_to_anchor=(0.0, 0.62))
b.plot(t, col(d, "v_filtrada"), color=AZUL, lw=1.6, label="velocidade filtrada")
b.step(t, col(d, "setpoint"), where="post", color=TINTA, ls="--", lw=1.2, label="setpoint")
b.set_ylabel("Velocidade (mm/s)")
b.set_xlabel("Tempo (s)")
b.legend(loc="lower left", fontsize=8)
fig.tight_layout()
salva(fig, "ensaio3_distancia")
