"""Figures for apresentacao.md (concept diagrams + charts from the real training logs).

    ~/ros2_ws/.venv/bin/python slides_figs.py      -> runs/slides_figs/*.png
"""
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "runs" / "slides_figs"
OUT.mkdir(parents=True, exist_ok=True)

# reference palette (light), validated: blue, orange, aqua, yellow
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
VIOLET = "#6250d6"
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
LIGHT_BLUE, LIGHT_ORANGE, LIGHT_AQUA, LIGHT_YELLOW = "#dbe8f8", "#fbe1d6", "#d3f0e5", "#fbecc8"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 13,
    "axes.titlesize": 16, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def box(ax, x, y, w, h, text, fc, ec, fs=14, weight="bold", color=INK):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, weight=weight, color=color, wrap=True)


def arrow(ax, p, q, text="", color=INK2, rad=0.0, fs=12, off=(0, 0.12)):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=22, lw=2.2, color=color,
                                 connectionstyle="arc3,rad=%.2f" % rad))
    if text:
        mx, my = (p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1]
        ax.text(mx, my, text, ha="center", va="center", fontsize=fs, color=INK2,
                bbox=dict(fc=SURF, ec="none", pad=2))


def blank(w=10, h=5.5):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_axis_off()
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    return fig, ax


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def smooth(y, k=15):
    y = np.asarray(y, float)
    return np.convolve(y, np.ones(k) / k, mode="valid")


# 1. the RL loop -----------------------------------------------------------------------
def fig_loop():
    fig, ax = blank(11, 5.6)
    box(ax, 2.6, 3.3, 3.4, 2.0, "AGENTE\n(rede neural\ndo robô)", LIGHT_BLUE, BLUE, fs=16)
    box(ax, 8.4, 3.3, 3.4, 2.0, "AMBIENTE\n(simulador:\nringue + adversário)", LIGHT_ORANGE, ORANGE, fs=16)
    arrow(ax, (4.3, 3.9), (6.7, 3.9), "AÇÃO  (v, ω)\n\"acelera e vira à esquerda\"", BLUE, rad=-0.35, off=(0, 0.75))
    arrow(ax, (6.7, 2.8), (4.3, 2.8), "ESTADO  s\n\"onde estou, onde está o rival\"", ORANGE, rad=-0.3, off=(0, -0.62))
    arrow(ax, (8.4, 2.25), (2.6, 2.25), "", AQUA, rad=-0.42)
    ax.text(5.5, 0.3, "RECOMPENSA  r   (+10 venceu · −10 caiu · −0,05 parado …)", ha="center", fontsize=14,
            color=INK, weight="bold", bbox=dict(fc=LIGHT_AQUA, ec=AQUA, boxstyle="round,pad=0.4"))
    ax.text(5.5, 5.35, "Repete 20 vezes por segundo, milhões de vezes", ha="center", fontsize=13, color=INK2)
    save(fig, "01_loop_rl.png")


# 2. reward as "petisco" table -----------------------------------------------------------------
def fig_rewards():
    terms = [("venceu (rival saiu do ringue)", 10), ("empate aos 40 s", -3), ("ficou parado (por passo)", -0.05),
             ("tempo passando (por passo, v5)", -0.01), ("caiu do ringue", -10)]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    names = [t[0] for t in terms][::-1]
    vals = [t[1] for t in terms][::-1]
    shown = [v if abs(v) >= 1 else (0.6 if v > 0 else -0.6) for v in vals]  # tiny terms drawn visible
    colors = [BLUE if v > 0 else ORANGE for v in vals]
    ax.barh(names, shown, color=colors, height=0.55)
    for i, (v, s) in enumerate(zip(vals, shown)):
        ax.text(s + (0.3 if s > 0 else -0.3), i, ("%+g" % v).replace(".", ","), va="center",
                ha="left" if s > 0 else "right", fontsize=14, weight="bold", color=INK)
    ax.axvline(0, color=INK2, lw=1.2)
    ax.set_xlim(-13, 13)
    ax.set_xticks([])
    ax.grid(False)
    ax.set_title("A recompensa é o \"petisco\" (ou a bronca) do robô — sumô")
    ax.text(12.8, -0.9, "termos por passo desenhados maiores para ficarem visíveis", ha="right", fontsize=10, color=MUTED)
    save(fig, "02_recompensa_sumo.png")


# 3. neural network drawing ----------------------------------------------------------------------
def fig_network():
    fig, ax = blank(12, 6.4)
    ins = ["minha posição (x, y)", "posição do rival em relação a mim", "rival em relação ao centro",
           "ângulo entre nós (cos, sen)", "minha velocidade", "velocidade do rival", "ação anterior"]
    yin = np.linspace(5.6, 0.8, len(ins))
    for y, t in zip(yin, ins):
        ax.add_patch(Circle((2.9, y), 0.17, fc=LIGHT_BLUE, ec=BLUE, lw=1.5))
        ax.text(2.55, y, t, ha="right", va="center", fontsize=11.5, color=INK)
    h1, h2 = np.linspace(5.8, 0.6, 9), np.linspace(5.8, 0.6, 9)
    for xs, ys in ((5.4, h1), (7.6, h2)):
        for y in ys:
            ax.add_patch(Circle((xs, y), 0.15, fc="#eeeeec", ec=MUTED, lw=1.2))
    for a in yin:
        for b in h1:
            ax.plot([3.07, 5.25], [a, b], color=GRID, lw=0.6, zorder=0)
    for a in h1:
        for b in h2:
            ax.plot([5.55, 7.45], [a, b], color=GRID, lw=0.6, zorder=0)
    outs = [(4.1, "v  (frente / ré)"), (2.3, "ω  (girar)")]
    for y, t in outs:
        for b in h2:
            ax.plot([7.75, 9.65], [b, y], color=GRID, lw=0.6, zorder=0)
        ax.add_patch(Circle((9.8, y), 0.2, fc=LIGHT_ORANGE, ec=ORANGE, lw=1.8))
        ax.text(10.15, y, t, va="center", fontsize=13, weight="bold")
    ax.text(2.9, 6.2, "12 entradas", ha="center", fontsize=13, weight="bold", color=BLUE)
    ax.text(5.4, 6.2, "64 neurônios", ha="center", fontsize=13, weight="bold", color=INK2)
    ax.text(7.6, 6.2, "64 neurônios", ha="center", fontsize=13, weight="bold", color=INK2)
    ax.text(9.8, 6.2, "2 saídas", ha="center", fontsize=13, weight="bold", color=ORANGE)
    ax.text(6.5, 0.05, "≈ 5 mil números (pesos) ajustados pelo treino · tudo calculado em numpy no robô",
            ha="center", fontsize=11.5, color=INK2)
    save(fig, "03_rede_neural.png")


# 4. actor-critic --------------------------------------------------------------------------------
def fig_actor_critic():
    fig, ax = blank(11, 5.2)
    box(ax, 2.4, 3.6, 3.6, 1.6, "ATOR  (o jogador)\ndecide a ação", LIGHT_BLUE, BLUE, fs=15)
    box(ax, 8.6, 3.6, 3.6, 1.6, "CRÍTICO  (o técnico)\n\"essa situação vale\nquantos pontos?\"", LIGHT_AQUA, AQUA, fs=14)
    box(ax, 5.5, 1.0, 5.0, 1.2, "Vantagem = o que aconteceu − o que o técnico esperava", LIGHT_YELLOW, YELLOW,
        fs=13, weight="normal")
    arrow(ax, (8.6, 2.75), (6.8, 1.65), "", AQUA)
    arrow(ax, (4.2, 1.65), (2.4, 2.75), "melhor que o esperado → repete\npior que o esperado → evita", INK2,
          off=(-1.3, 0.25))
    ax.text(2.4, 4.85, "vai para o robô", ha="center", fontsize=12, color=BLUE, weight="bold")
    ax.text(8.6, 4.85, "só existe no treino", ha="center", fontsize=12, color=AQUA, weight="bold")
    save(fig, "04_ator_critico.png")


# 5. trial and error: learning curve, with vs without shaping (real data) ----------------------------
def fig_learning_curve():
    a = read(HERE / "runs/sumo_ppo_v1/log.csv")
    b = read(HERE / "runs/sumo_ppo_noshaping/log.csv")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for rows, col, lab in ((a, BLUE, "com dica (shaping)"), (b, ORANGE, "sem dica: só +10 ao vencer")):
        x = np.array([float(r["steps"]) for r in rows]) / 1e6
        y = smooth([100 * float(r["win"]) for r in rows], 9)
        ax.plot(x[len(x) - len(y):], y, color=col, lw=2.4)
        ax.plot([], [], color=col, lw=2.4, label=lab)
    ax.annotate("aprende em ~2,5 M de passos", xy=(2.5, 95), xytext=(6, 60), fontsize=12, color=INK2,
                arrowprops=dict(arrowstyle="->", color=INK2))
    ax.annotate("passa ~13 M de passos\nandando sem empurrar", xy=(12, 3), xytext=(24, 35), fontsize=12, color=INK2,
                arrowprops=dict(arrowstyle="->", color=INK2))
    ax.set_xlabel("passos de treino (milhões)")
    ax.set_ylabel("vitórias (%)")
    ax.set_ylim(-3, 105)
    ax.set_xlim(0, 34)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Tentativa e erro: vitórias contra um robô parado (fase 1)")
    save(fig, "05_curva_aprendizado.png")


# 6. shaping potential heatmap ------------------------------------------------------------------
def fig_shaping():
    R = 0.40
    me = np.array([-0.05, 0.0])
    xs = np.linspace(-R, R, 301)
    X, Y = np.meshgrid(xs, xs)
    phi = -1.0 * np.hypot(X - me[0], Y - me[1]) + 2.0 * np.hypot(X, Y)
    phi[np.hypot(X, Y) > R] = np.nan
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    im = ax.imshow(phi, extent=(-R, R, -R, R), origin="lower", cmap="Blues")
    ax.add_patch(Circle((0, 0), R, fill=False, ec=INK2, lw=2))
    ax.add_patch(Circle(tuple(me), 0.037, fc=BLUE, ec="white", lw=2))
    ax.text(me[0], me[1] - 0.075, "eu", ha="center", fontsize=13, weight="bold", color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.75)
    cb.set_label("\"pontos de dica\" se o rival estiver aqui")
    ax.set_title("Dica de recompensa: rival perto da borda = bom")
    save(fig, "06_shaping_mapa.png")


# 7. parallel simulation ------------------------------------------------------------------------
def fig_parallel():
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_axes([0.0, 0.0, 0.5, 1.0])
    ax.set_axis_off()
    rng = np.random.default_rng(1)
    for i in range(8):
        for j in range(8):
            cx, cy = j * 1.0, -i * 1.0
            ax.add_patch(Circle((cx, cy), 0.42, fc="#2b2b2b", ec="white", lw=1))
            for col in (BLUE, ORANGE):
                r, a = 0.3 * rng.random() ** 0.5, 2 * math.pi * rng.random()
                ax.add_patch(Circle((cx + r * math.cos(a), cy + r * math.sin(a)), 0.07, fc=col, ec="none"))
    ax.set_xlim(-0.6, 7.6)
    ax.set_ylim(-7.6, 0.9)
    ax.set_aspect("equal")
    ax.text(3.5, 0.7, "64 de 2048 ringues simulados ao mesmo tempo na GPU", ha="center", fontsize=12, color=INK2)
    bx = fig.add_axes([0.66, 0.18, 0.31, 0.62])
    vals = [950, 200_000]
    bx.barh(["simulador normal\n(1 ringue)", "treino paralelo\n(GPU)"], vals, color=[ORANGE, BLUE], height=0.5)
    for i, v in enumerate(vals):
        bx.text(v + 3000, i, "%s passos/s" % format(v, ",").replace(",", "."), va="center", fontsize=13, weight="bold")
    bx.set_xlim(0, 330_000)
    bx.set_xticks([])
    bx.grid(False)
    bx.set_title("≈ 200× mais rápido")
    bx.text(0, -0.75, "1 hora de treino ≈ 9 dias de simulação normal", fontsize=12, color=INK2, transform=bx.transData)
    save(fig, "07_simulacao_paralela.png")


# 8. curriculum staircase -----------------------------------------------------------------------
def fig_curriculum():
    fig, ax = blank(12, 6.4)
    steps = [("Fase 1", "robô parado", "aprende a\nempurrar", "33 M · 3 min"),
             ("Fase 2", "5 rivais com script", "aprende a\nnão bater de frente", "52 M · 6 min"),
             ("Fase 3", "cópias de si mesmo", "ataque de lado,\nborda mais próxima", "98 M · 13 min"),
             ("Fase 4", "perseguidores\ninsistentes", "não aceitar\no empate", "54 M · ~20 min"),
             ("Fase 5", "perseguidores fortes\n+ clinch", "sair do\ncorpo a corpo", "79 M · 32 min")]
    cols = [LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, LIGHT_ORANGE, LIGHT_ORANGE]
    ecs = [BLUE, BLUE, BLUE, ORANGE, ORANGE]
    for k, (t, opp, learn, cost) in enumerate(steps):
        x, h = 0.3 + k * 2.35, 1.9 + 0.6 * k
        ax.add_patch(FancyBboxPatch((x, 0.3), 2.2, h, boxstyle="round,pad=0.0,rounding_size=0.08", fc=cols[k], ec=ecs[k], lw=2))
        ax.text(x + 1.1, 0.3 + h - 0.25, t, ha="center", va="top", fontsize=15, weight="bold")
        ax.text(x + 1.1, 0.3 + h + 0.55, opp, ha="center", va="center", fontsize=11.5, color=INK2)
        ax.text(x + 1.1, 0.75, learn, ha="center", va="center", fontsize=11, color=INK)
        ax.text(x + 1.1, 0.3 + h - 0.62, cost, ha="center", va="top", fontsize=10.5, color=MUTED)
    ax.text(3.8, 6.15, "v1 → v3  (entrega antiga)", ha="center", fontsize=13, color=BLUE, weight="bold")
    ax.text(9.6, 6.15, "v4 → v5  (entrega nova)", ha="center", fontsize=13, color=ORANGE, weight="bold")
    save(fig, "08_curriculo.png")


# 9. self-play pool ---------------------------------------------------------------------------
def fig_selfplay():
    fig, ax = blank(11, 4.6)
    box(ax, 1.8, 2.3, 2.6, 1.6, "rede que\nestá aprendendo", LIGHT_BLUE, BLUE, fs=14)
    names = ["it 100", "it 300", "it 500", "…", "it 1400", "mais\nrecente"]
    for k, n in enumerate(names):
        x = 4.7 + k * 1.05
        ax.add_patch(FancyBboxPatch((x - 0.45, 1.75), 0.9, 1.1, boxstyle="round,pad=0.0,rounding_size=0.06",
                                    fc="#eeeeec" if n != "…" else SURF, ec=MUTED if n != "…" else SURF, lw=1.5))
        ax.text(x, 2.3, n, ha="center", va="center", fontsize=10.5, color=INK2)
    ax.text(7.3, 3.25, "cópias congeladas (não aprendem)", ha="center", fontsize=12.5, weight="bold", color=INK2)
    arrow(ax, (3.15, 2.3), (4.15, 2.3), "", INK2)
    ax.text(7.3, 1.2, "a cada 100 iterações entra uma cópia nova · cada rodada sorteia um rival", ha="center",
            fontsize=12, color=INK2)
    ax.text(7.3, 0.6, "+ 30% de rivais com script, para não esquecer como vencê-los", ha="center", fontsize=12, color=INK2)
    save(fig, "09_self_play.png")


# 10. discount: why the draw penalty did not reach the network ---------------------------------------
def fig_discount():
    t = np.arange(0, 801)
    w = 0.99 ** t
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(t * 0.05, 100 * w, color=BLUE, lw=2.4)
    ax.set_xlabel("quando o evento acontece (segundos à frente)")
    ax.set_ylabel("quanto ele \"pesa\" agora (%)")
    for s, lab in ((1, "1 s: 82%"), (5, "5 s: 37%"), (40, "40 s: 0,03%")):
        y = 100 * 0.99 ** (s / 0.05)
        ax.plot([s], [y], "o", color=BLUE, ms=9, mec="white", mew=2)
        ax.annotate(lab, (s, y), xytext=(10, 12), textcoords="offset points", fontsize=12.5, weight="bold")
    ax.annotate("a penalidade de empate (−3)\nsó chega aos 40 s → vale quase nada", xy=(40, 0.5), xytext=(17, 45),
                fontsize=12.5, color=INK2, arrowprops=dict(arrowstyle="->", color=INK2))
    ax.set_ylim(-3, 105)
    ax.set_title("Desconto γ = 0,99: o futuro distante quase não conta")
    save(fig, "10_desconto.png")


# 11. clinch schematic ---------------------------------------------------------------------------
def fig_clinch():
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax in axs:
        ax.set_xlim(-0.45, 0.45)
        ax.set_ylim(-0.45, 0.45)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.add_patch(Circle((0, 0), 0.40, fc="#2b2b2b", ec="white", lw=3))
    a = axs[0]
    a.add_patch(Circle((-0.04, 0.05), 0.037, fc=BLUE, ec="white", lw=2))
    a.add_patch(Circle((0.034, 0.05), 0.037, fc=ORANGE, ec="white", lw=2))
    a.annotate("", xy=(-0.12, 0.05), xytext=(-0.02, 0.05), arrowprops=dict(arrowstyle="-|>", color="white", lw=3))
    a.annotate("", xy=(-0.0, 0.05), xytext=(0.11, 0.05), arrowprops=dict(arrowstyle="-|>", color=YELLOW, lw=3))
    a.text(0, -0.17, "0,20 N  ×  0,20 N\nforças iguais → ninguém sai", ha="center", color="white", fontsize=13, weight="bold")
    a.set_title("Clinch: empurrão de frente", loc="center")
    b = axs[1]
    b.add_patch(Circle((0.22, 0.12), 0.037, fc=ORANGE, ec="white", lw=2))
    b.add_patch(Circle((0.16, 0.05), 0.037, fc=BLUE, ec="white", lw=2))
    b.annotate("", xy=(0.36, 0.22), xytext=(0.17, 0.06), arrowprops=dict(arrowstyle="-|>", color="white", lw=3))
    b.plot([0.0, 0.16], [-0.12, 0.05], ls=":", color=BLUE, lw=2.5)
    b.text(0, -0.2, "sai pelo lado e empurra\nrumo à borda mais próxima", ha="center", color="white", fontsize=13, weight="bold")
    b.set_title("O que vence: ângulo + borda", loc="center")
    save(fig, "11_clinch.png")


# 12. phase 5 training curves (real) ----------------------------------------------------------------
def fig_v5_training():
    rows = read(HERE / "runs/sumo_ppo_v5/log.csv")
    x = np.array([float(r["steps"]) for r in rows]) / 1e6
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ends = []
    for key, col, lab in (("win_pursuer", BLUE, "vs perseguidor"), ("win_pursuer_bi", AQUA, "vs perseguidor de ré"),
                          ("win_self", ORANGE, "vs cópias de si mesmo")):
        y = smooth([100 * float(r[key]) for r in rows], 41)
        ax.plot(x[len(x) - len(y):], y, color=col, lw=2.2)
        ends.append([y[-1], lab])
    ends.sort()
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + 5)
    for yv, lab in ends:
        ax.text(x[-1] + 1, yv, lab, va="center", fontsize=12)
    ax.set_xlim(0, 95)
    ax.set_ylim(0, 100)
    ax.set_xlabel("passos de treino na fase 5 (milhões)")
    ax.set_ylabel("vitórias no treino (%)")
    ax.set_title("Fase 5: vitórias durante o treino (média móvel)")
    save(fig, "12_treino_v5.png")


# 13. v3 vs v5 results ---------------------------------------------------------------------------
def fig_results_sumo():
    opp = ["perseguidor", "perseguidor\nde ré", "chaser", "careful", "evader"]
    v3 = [36.3, 41.6, 88.9, 89.4, 88.4]
    v5 = [69, 66, 91, 100, 97]
    x = np.arange(len(opp))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - 0.2, v3, 0.38, color=ORANGE, label="v3 (antiga)")
    ax.bar(x + 0.2, v5, 0.38, color=BLUE, label="v5 (nova)")
    for i in range(len(opp)):
        ax.text(x[i] - 0.2, v3[i] + 1.5, "%.0f" % v3[i], ha="center", fontsize=11.5)
        ax.text(x[i] + 0.2, v5[i] + 1.5, "%.0f" % v5[i], ha="center", fontsize=11.5, weight="bold")
    ax.set_xticks(x, opp)
    ax.set_ylim(0, 110)
    ax.set_ylabel("vitórias (%)")
    ax.legend(frameon=False, loc="upper left", ncols=2)
    ax.set_title("Sumô: vitórias por adversário (512 rodadas cada, simulação vetorizada)")
    save(fig, "13_resultados_v3_v5.png")


# 14. what makes the pursuer unbeatable --------------------------------------------------------------
def fig_sweep():
    labs = ["pior caso: 100% vel., KW 5,\nmira 0,3 s, sem guarda", "KW 3 (gira mais devagar)", "sem mirar à frente",
            "com guarda de borda", "velocidade 85%"]
    vals = [0.0, 72.3, 87.7, 98.8, 99.4]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.barh(labs[::-1], vals[::-1], color=[BLUE] * 4 + [ORANGE], height=0.55)
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 1.2, i, ("%.1f%%" % v).replace(".", ",") if v else "0% (empata)", va="center", fontsize=12.5,
                weight="bold")
    ax.set_xlim(0, 115)
    ax.set_xticks([])
    ax.grid(False)
    ax.set_title("v5 contra o perseguidor: basta ele ser um pouco mais fraco")
    save(fig, "14_perseguidor_variacoes.png")


# 15. soccer roles + results -------------------------------------------------------------------------
def fig_soccer():
    fig, ax = plt.subplots(figsize=(11, 6))
    L, W, G = 0.9, 0.6, 0.2
    ax.add_patch(plt.Rectangle((-L, -W), 2 * L, 2 * W, fc="#e8f1e4", ec=INK2, lw=2))
    ax.plot([0, 0], [-W, W], color=INK2, lw=1.5)
    ax.add_patch(Circle((0, 0), 0.15, fill=False, ec=INK2, lw=1.5))
    ax.plot([-L, -L], [-G, G], color=BLUE, lw=8)
    ax.plot([L, L], [-G, G], color=ORANGE, lw=8)
    ax.add_patch(Circle((0.25, 0.1), 0.025, fc="white", ec=INK, lw=1.5))
    robots = [((-0.78, 0.0), "goleiro\n(script)", BLUE), ((0.15, 0.32), "atacante 1\n(rede)", BLUE),
              ((0.05, -0.2), "atacante 2\n(mesma rede)", BLUE)]
    for (x, y), t, c in robots:
        ax.add_patch(Circle((x, y), 0.04, fc=c, ec="white", lw=2))
        ax.text(x, y - 0.1, t, ha="center", va="top", fontsize=12, weight="bold")
    for (x, y) in ((0.75, 0.0), (0.45, 0.25), (0.5, -0.15)):
        ax.add_patch(Circle((x, y), 0.04, fc=ORANGE, ec="white", lw=2))
    ax.annotate("", xy=(0.88, 0.05), xytext=(0.29, 0.11), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=2, ls="--"))
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.7, 0.7)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Futebol 3×3: um goleiro com regras + dois atacantes com a MESMA rede neural")
    save(fig, "15_futebol_papeis.png")

    fig, ax = plt.subplots(figsize=(10, 4.4))
    names = ["A2", "A3 (self-play)", "A2w (final)", "A2w + supervisor"]
    gf, gs = [4, 0, 10, 9], [1, 0, 1, 2]
    x = np.arange(len(names))
    ax.bar(x - 0.2, gf, 0.38, color=BLUE, label="gols feitos")
    ax.bar(x + 0.2, gs, 0.38, color=ORANGE, label="gols sofridos")
    for i in range(len(names)):
        ax.text(x[i] - 0.2, gf[i] + 0.2, str(gf[i]), ha="center", fontsize=12, weight="bold")
        ax.text(x[i] + 0.2, gs[i] + 0.2, str(gs[i]), ha="center", fontsize=12)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 12)
    ax.set_ylabel("gols em 3 partidas de 5 min")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Futebol: versões de RL contra o time com script")
    save(fig, "16_futebol_resultados.png")


# 16. CTF: arena, roles and the avoidance field ----------------------------------------------------------
def fig_ctf():
    fig, ax = plt.subplots(figsize=(11, 6.6))
    H = 0.8
    ax.add_patch(plt.Rectangle((-H, -H), H, 2 * H, fc=LIGHT_BLUE, ec="none"))
    ax.add_patch(plt.Rectangle((0, -H), H, 2 * H, fc=LIGHT_ORANGE, ec="none"))
    ax.add_patch(plt.Rectangle((-H, -H), 2 * H, 2 * H, fill=False, ec=INK2, lw=2))
    ax.plot([0, 0], [-H, H], color=INK2, lw=1.5, ls="--")
    for x, c in ((-0.62, BLUE), (0.62, ORANGE)):
        ax.add_patch(Circle((x, 0), 0.14, fill=False, ec=c, lw=2))
        ax.plot([x], [0], marker="^", ms=14, color=c)
    # avoidance field around the enemy guard (only in the enemy half)
    ox, oy, gx, gy = 0.38, 0.05, 0.62, 0.0
    xs, ys = np.meshgrid(np.linspace(0.05, 0.78, 13), np.linspace(-0.6, 0.6, 13))
    U, V = np.zeros_like(xs), np.zeros_like(ys)
    for i in range(xs.shape[0]):
        for j in range(xs.shape[1]):
            x, y = xs[i, j], ys[i, j]
            d = math.hypot(gx - x, gy - y) + 1e-6
            fx, fy = (gx - x) / d, (gy - y) / d
            f0 = (fx, fy)
            dist = math.hypot(x - ox, y - oy)
            if 1e-6 < dist < 0.30:
                force = 0.30 / dist - 1.0
                rx, ry = (x - ox) / dist, (y - oy) / dist
                tx, ty = -ry, rx
                dot = tx * f0[0] + ty * f0[1]
                if abs(dot) < 0.2:
                    if ty * (1 if y - oy >= 0 else -1) < 0:
                        tx, ty = -tx, -ty
                elif dot < 0:
                    tx, ty = -tx, -ty
                fx += force * (rx + 2.0 * tx)
                fy += force * (ry + 2.0 * ty)
            n = math.hypot(fx, fy) + 1e-9
            U[i, j], V[i, j] = fx / n, fy / n
    ax.quiver(xs, ys, U, V, color=MUTED, scale=28, width=0.003)
    ax.add_patch(Circle((ox, oy), 0.30, fill=False, ec=ORANGE, lw=1.5, ls=":"))
    ax.add_patch(Circle((ox, oy), 0.04, fc=ORANGE, ec="white", lw=2))
    ax.text(ox, oy - 0.36, "zona de desvio (30 cm)", ha="center", fontsize=11, color=INK2,
            bbox=dict(fc=LIGHT_ORANGE, ec="none", pad=1))
    ax.add_patch(Circle((-0.52, 0.12), 0.04, fc=BLUE, ec="white", lw=2))
    ax.text(-0.52, 0.22, "defensor\n(goleiro a 10 cm)", ha="center", fontsize=11.5, weight="bold")
    ax.add_patch(Circle((0.08, -0.35), 0.04, fc=BLUE, ec="white", lw=2))
    ax.text(0.08, -0.47, "atacante", ha="center", fontsize=11.5, weight="bold")
    ax.text(-0.4, -0.73, "meu campo:\nposso SEGURAR", ha="center", fontsize=11, color=BLUE, weight="bold")
    ax.text(0.4, -0.73, "campo inimigo:\nposso SER SEGURADO", ha="center", fontsize=11, color=ORANGE, weight="bold")
    ax.set_xlim(-0.85, 0.85)
    ax.set_ylim(-0.85, 0.85)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Rouba-bandeira: campo potencial (setas = direção que o atacante segue)")
    save(fig, "17_ctf_campo.png")


def fig_ctf_states():
    fig, ax = blank(12, 5.0)
    box(ax, 2.0, 3.6, 3.0, 1.2, "ATACANTE\nsem bandeira", LIGHT_BLUE, BLUE, fs=13)
    box(ax, 6.0, 3.6, 3.0, 1.2, "vai à bandeira\ndesviando (avoid)", "#eeeeec", MUTED, fs=12, weight="normal")
    box(ax, 10.0, 3.6, 3.0, 1.2, "guarda em cima\nda bandeira (< 8 cm)?\n→ vai direto", "#eeeeec", MUTED, fs=11.5,
        weight="normal")
    box(ax, 2.0, 1.2, 3.0, 1.2, "ATACANTE\ncom bandeira", LIGHT_BLUE, BLUE, fs=13)
    box(ax, 6.0, 1.2, 3.0, 1.2, "volta para a base\ndesviando", "#eeeeec", MUTED, fs=12, weight="normal")
    box(ax, 10.0, 1.2, 3.0, 1.2, "DEFENSOR: goleiro a 10 cm;\nbandeira roubada →\npersegue o ladrão", LIGHT_AQUA, AQUA,
        fs=11.5, weight="normal")
    arrow(ax, (3.5, 3.6), (4.5, 3.6))
    arrow(ax, (7.5, 3.6), (8.5, 3.6))
    arrow(ax, (3.5, 1.2), (4.5, 1.2))
    arrow(ax, (2.0, 3.0), (2.0, 1.8), "pegou", INK2, off=(0.5, 0))
    arrow(ax, (1.3, 1.8), (1.3, 3.0), "pontuou /\nfoi segurado", INK2, off=(-0.85, 0))
    save(fig, "18_ctf_estados.png")


def fig_ctf_old_new():
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    H = 0.8
    for ax, title in zip(axs, ("ANTIGA: finta na volta", "NOVA: volta direta + desvio")):
        ax.add_patch(plt.Rectangle((-H, -H), H, 2 * H, fc=LIGHT_BLUE, ec="none"))
        ax.add_patch(plt.Rectangle((0, -H), H, 2 * H, fc=LIGHT_ORANGE, ec="none"))
        ax.plot([0, 0], [-H, H], color=INK2, lw=1.5, ls="--")
        ax.add_patch(Circle((-0.62, 0), 0.14, fill=False, ec=BLUE, lw=2))
        ax.add_patch(Circle((0.55, -0.15), 0.04, fc=BLUE, ec="white", lw=2))
        ax.set_xlim(-H, H)
        ax.set_ylim(-H, H)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(title, loc="center")
    a = axs[0]
    a.annotate("", xy=(0.02, 0.5), xytext=(0.52, -0.12), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.5, ls="--"))
    a.annotate("", xy=(0.02, -0.5), xytext=(0.06, 0.45), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.5))
    a.text(0.35, 0.55, "0,6 s no\nlado falso", fontsize=11.5, ha="center")
    a.text(-0.35, -0.55, "alvo em x = 0:\npode parar na linha", fontsize=11.5, ha="center", color=INK2)
    b = axs[1]
    t = np.linspace(0, 1, 50)
    px = 0.55 + (-0.62 - 0.55) * t
    py = -0.15 + 0.15 * t - 0.25 * np.sin(math.pi * t)
    b.plot(px, py, color=BLUE, lw=2.5)
    b.annotate("", xy=(-0.6, -0.01), xytext=(-0.55, -0.03), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.5))
    b.add_patch(Circle((0.2, 0.0), 0.04, fc=ORANGE, ec="white", lw=2))
    b.text(0.2, 0.1, "inimigo", ha="center", fontsize=11.5)
    b.text(-0.1, -0.6, "contorna com K_TAN = 2,0", fontsize=11.5, ha="center", color=INK2)
    save(fig, "19_ctf_antiga_nova.png")


if __name__ == "__main__":
  for f in (fig_loop, fig_rewards, fig_network, fig_actor_critic, fig_learning_curve, fig_shaping, fig_parallel,
          fig_curriculum, fig_selfplay, fig_discount, fig_clinch, fig_v5_training, fig_results_sumo, fig_sweep,
          fig_soccer, fig_ctf, fig_ctf_states, fig_ctf_old_new):
    f()
