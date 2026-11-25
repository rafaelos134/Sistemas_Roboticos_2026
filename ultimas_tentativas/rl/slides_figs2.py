"""More figures for apresentacao.md: training phases and reward ("petisco") tables of soccer and CTF.

    ~/ros2_ws/.venv/bin/python slides_figs2.py      -> runs/slides_figs/2*.png
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from slides_figs import (AQUA, BLUE, INK, INK2, LIGHT_AQUA, LIGHT_BLUE, LIGHT_ORANGE, MUTED, ORANGE, arrow, blank,
                         save)

GRAY_FC = "#eeeeec"


def card(ax, x, y, w, h, title, lines, fc, ec, cost="", dy=-0.25):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.0,rounding_size=0.08",
                                fc=fc, ec=ec, lw=2))
    ax.text(x, y + h / 2 - 0.22, title, ha="center", va="top", fontsize=15, weight="bold", color=INK)
    if cost:
        ax.text(x, y + h / 2 - 0.62, cost, ha="center", va="top", fontsize=10.5, color=MUTED)
    ax.text(x, y + dy, lines, ha="center", va="center", fontsize=11.5, color=INK, linespacing=1.4)


def reward_panel(ax, title, terms, scale=15):
    """terms: (label, value, per_step). Per-step terms are drawn bigger so they are visible."""
    names = [t[0] for t in terms][::-1]
    vals = [t[1] for t in terms][::-1]
    per = [t[2] for t in terms][::-1]
    shown = [v if abs(v) >= 1 else (0.7 if v > 0 else -0.7) for v in vals]
    ax.barh(names, shown, color=[BLUE if v > 0 else ORANGE for v in vals], height=0.55)
    for i, (v, sv, ps) in enumerate(zip(vals, shown, per)):
        lab = ("%+g" % v).replace(".", ",") + (" /passo" if ps else "")
        ax.text(sv + (0.3 if sv > 0 else -0.3), i, lab, va="center", ha="left" if sv > 0 else "right",
                fontsize=12.5, weight="bold", color=INK)
    ax.axvline(0, color=INK2, lw=1.2)
    ax.set_xlim(-scale, scale)
    ax.set_xticks([])
    ax.grid(False)
    ax.set_title(title)


# --- soccer: curriculum --------------------------------------------------------------------------
def fig_soccer_curriculum():
    fig, ax = blank(13, 6.8)
    card(ax, 1.8, 3.9, 3.1, 2.6, "A1", "rival: time PARADO\n\naprende a levar\na bola ao gol", LIGHT_BLUE, BLUE,
         "600 it · 39 M · 7 min")
    card(ax, 5.2, 3.9, 3.1, 2.6, "A2", "rival: time com SCRIPT\n(ganhos sorteados)\n\naprende a disputar\na bola",
         LIGHT_BLUE, BLUE, "3 × 1000 it · ≈ 197 M")
    card(ax, 8.6, 3.9, 3.1, 2.6, "A2w  ✔ final", "igual ao A2 +\n\"bola longe das paredes\"\n\nmenos travamento",
         LIGHT_BLUE, BLUE, "1000 it · ≈ 66 M · 14 min")
    card(ax, 11.6, 5.15, 2.7, 2.3, "A3", "self-play:\ncópias de si mesmo\n→ piorou", GRAY_FC, MUTED,
         "1500 it · ≈ 98 M · 28 min", dy=-0.4)
    card(ax, 11.6, 2.45, 2.7, 2.3, "D", "defensor de RL\nvs nossos atacantes\n→ empatou c/ script", GRAY_FC, MUTED,
         "800 it · ≈ 26 M · 20 min", dy=-0.4)
    arrow(ax, (3.35, 3.9), (3.65, 3.9))
    arrow(ax, (6.75, 3.9), (7.05, 3.9))
    arrow(ax, (10.15, 4.4), (10.25, 4.9))
    arrow(ax, (10.15, 3.4), (10.25, 2.8))
    ax.text(5.2, 0.75, "Atacantes: uma rede compartilhada pelos dois robôs · goleiro final: com regras (script)",
            ha="center", fontsize=12.5, color=INK2)
    ax.text(5.2, 0.3, "M = milhões de passos (\"≈\": calculado por iterações × 32 × 1024 × robôs)", ha="center",
            fontsize=10.5, color=MUTED)
    ax.set_title("Futebol: currículo de treino", loc="left", fontsize=17, weight="bold")
    save(fig, "21_futebol_curriculo.png")


# --- soccer: rewards ---------------------------------------------------------------------------
def fig_soccer_rewards():
    fig, axs = plt.subplots(1, 2, figsize=(16, 4.8), gridspec_kw={"wspace": 0.55})
    reward_panel(axs[0], "Atacantes", [
        ("gol feito", 10, False), ("gol sofrido", -10, False), ("atacante parado", -0.02, True),
        ("bola travada (< 3 cm em 2 s)", -0.05, True)])
    reward_panel(axs[1], "Defensor (treinado, não usado)", [
        ("afastar a bola do nosso gol", 0.5, True), ("gol sofrido", -10, False),
        ("sair da zona defensiva", -0.05, True), ("bola travada no nosso campo", -0.05, True)])
    fig.suptitle("Futebol: o \"petisco\" de cada robô", x=0.02, y=1.04, ha="left", fontsize=17, weight="bold")
    fig.text(0.02, -0.06, "Dicas (shaping) dos atacantes: bola mais perto do gol deles (×3), atacante mais perto da "
             "bola (×1) e bola longe das paredes laterais (×2, versão A2w).   /passo = a cada 0,05 s, "
             "barras pequenas desenhadas maiores.", fontsize=11, color=INK2, wrap=True)
    save(fig, "22_futebol_recompensa.png")


# --- CTF: league (alternating training) --------------------------------------------------------------
def fig_ctf_curriculum():
    fig, ax = blank(13.5, 6.0)
    att = [("600 it", "39 M", "vs scripts"), ("600 it", "39 M", ""), ("600 it", "39 M", ""), ("600 it", "39 M", ""),
           ("1000 it", "66 M", "+ rivais fortes")]
    dfn = [("400 it", "26 M", ""), ("400 it", "26 M", ""), ("400 it", "26 M", ""), ("600 it", "39 M", ""),
           ("800 it", "52 M", "+ rivais fortes")]
    for k in range(5):
        x = 2.2 + k * 2.45
        ax.text(x, 5.55, "rodada %d" % (k + 1), ha="center", fontsize=13, weight="bold", color=INK2)
        for (y, (it, st, extra), fc, ec, lab) in ((4.1, att[k], LIGHT_BLUE, BLUE, "ATACANTE"),
                                                   (1.6, dfn[k], LIGHT_AQUA, AQUA, "DEFENSOR")):
            ax.add_patch(FancyBboxPatch((x - 1.0, y - 0.75), 2.0, 1.5, boxstyle="round,pad=0.0,rounding_size=0.08",
                                        fc=fc, ec=ec, lw=2))
            ax.text(x, y + 0.38, lab, ha="center", fontsize=12.5, weight="bold")
            ax.text(x, y - 0.05, "%s · %s" % (it, st), ha="center", fontsize=11, color=INK2)
            if extra:
                ax.text(x, y - 0.42, extra, ha="center", fontsize=10.5, color=INK2)
        arrow(ax, (x, 3.35), (x, 2.35), "", MUTED)
        if k < 4:
            arrow(ax, (x + 0.2, 2.35), (x + 2.25, 3.35), "", MUTED)
    ax.text(0.15, 4.1, "aprende\na atacar", ha="left", va="center", fontsize=11.5, color=BLUE, weight="bold")
    ax.text(0.15, 1.6, "aprende\na defender", ha="left", va="center", fontsize=11.5, color=AQUA, weight="bold")
    ax.text(6.9, 0.35, "Cada um treina contra as versões CONGELADAS do outro papel (setas) · total ≈ 390 M de passos",
            ha="center", fontsize=12, color=INK2)
    ax.set_title("Rouba-bandeira com RL (experimento): liga alternada atacante × defensor", loc="left",
                 fontsize=16, weight="bold")
    save(fig, "23_ctf_curriculo.png")


# --- CTF: rewards ---------------------------------------------------------------------------------
def fig_ctf_rewards():
    fig, axs = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={"wspace": 0.75})
    reward_panel(axs[0], "Atacante", [
        ("nosso time pontuou", 10, False), ("peguei a bandeira inimiga", 3, False),
        ("peguei enquanto imune (extra)", 1, False), ("segurei um invasor", 1, False),
        ("fui segurado", -1, False), ("fui segurado COM a bandeira (extra)", -2, False),
        ("o outro time pontuou", -5, False)])
    reward_panel(axs[1], "Defensor", [
        ("segurei o ladrão da bandeira (extra)", 3, False), ("segurei um invasor", 2, False),
        ("nosso time pontuou", 2, False), ("fiquei no campo inimigo", -0.01, True), ("fui segurado", -2, False),
        ("inimigo pegou nossa bandeira", -3, False), ("o outro time pontuou", -10, False)])
    fig.suptitle("Rouba-bandeira com RL: o \"petisco\" de cada papel", x=0.02, y=1.04, ha="left", fontsize=17, weight="bold")
    fig.text(0.02, -0.05, "Dicas (shaping): atacante → chegar perto da bandeira e, com ela, perto da base; manter "
             "18 cm dos defensores no campo deles. Defensor → ficar no ponto de guarda (10 cm da bandeira) "
             "ou perto do ladrão.", fontsize=11, color=INK2, wrap=True)
    save(fig, "24_ctf_recompensa.png")


if __name__ == "__main__":
    for f in (fig_soccer_curriculum, fig_soccer_rewards, fig_ctf_curriculum, fig_ctf_rewards):
        f()
