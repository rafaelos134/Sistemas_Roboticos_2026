#!/usr/bin/env python3
"""Gráficos e métricas do Relatório 5 (exercícios avoid e goto).

Cada figura é salva em UM arquivo, quadrada e no mesmo estilo visual
(contorno completo, grade pontilhada, legenda com moldura, sem título na
imagem — o título fica na legenda do LaTeX). Para cada exercício há dois
ensaios, feitos com conjuntos de parâmetros diferentes (Config. 1 e Config. 2).
Os parâmetros de cada ensaio estão em CONFIG_AVOID e CONFIG_GOTO e foram
copiados de avoid.py e goto.py.

Uso (da pasta do projeto, fora do container):

    python gerar_graficos.py
    python gerar_graficos.py --formatos png
    python gerar_graficos.py --avoid a.csv b.csv --goto c.csv d.csv

Por padrão os logs são cortados na duração nominal de cada ensaio (300 s no
avoid e 180 s no goto) para que as duas configurações sejam comparadas na
mesma janela de tempo.

Saída (pasta graficos/), um arquivo por figura:
    avoid_trajetoria_cfg1 / _cfg2   trajetória na arena
    avoid_dmin_cfg1 / _cfg2         menor distância do lidar ao longo do tempo
    avoid_hist_dmin                 distribuição da menor distância (comparação)
    avoid_hist_omega                distribuição do comando angular (comparação)
    avoid_vel_dmin                  velocidade linear × menor distância (comparação)
    goto_trajetoria_cfg1 / _cfg2    trajetória com as metas atingidas
    goto_rho                        ρ ao longo do tempo (duas configurações)
    goto_alpha                      α ao longo do tempo (duas configurações)
    goto_vel_rho                    velocidade linear perto da meta (comparação)
    goto_alpha_resp                 erro de orientação após nova meta (comparação)
    goto_tempo_rho0                 tempo até a meta × distância inicial (comparação)
    goto_tempo_meta                 tempo por meta atingida (comparação)
    resumo.md                       tabelas com as métricas pedidas no enunciado
    goto_metas.csv                  dados de cada meta atingida

Dependências: numpy e matplotlib.
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

AQUI = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Ensaios e parâmetros (ordem: 1ª simulação, 2ª simulação)
# --------------------------------------------------------------------------
CONFIG_AVOID = [
    {"csv": "exercicios/avoid/avoid_log_20261103_224503.csv", "nome": "Config. 1",
     "params": {"K_V": 0.6, "K_W": 8.0, "D_SLOW": 0.25}},
    {"csv": "exercicios/avoid/avoid_log_20261103_230318.csv", "nome": "Config. 2",
     "params": {"K_V": 1.0, "K_W": 4.0, "D_SLOW": 0.145}},
]
CONFIG_GOTO = [
    {"csv": "exercicios/goto/goto_log_20261103_231411.csv", "nome": "Config. 1",
     "params": {"K_RHO": 1.0, "K_ALPHA": 4.0}},
    {"csv": "exercicios/goto/goto_log_20261103_232054.csv", "nome": "Config. 2",
     "params": {"K_RHO": 3.0, "K_ALPHA": 6.0}},
]

# nome no código -> (rótulo mathtext, rótulo em texto puro, unidade)
ROTULOS = {
    "K_V": (r"$K_V$", "K_V", ""),
    "K_W": (r"$K_W$", "K_W", ""),
    "D_SLOW": (r"$D_{\mathrm{slow}}$", "D_SLOW", " m"),
    "K_RHO": (r"$K_\rho$", "K_RHO", ""),
    "K_ALPHA": (r"$K_\alpha$", "K_ALPHA", ""),
}

# --------------------------------------------------------------------------
# Constantes do simulador (psr_hero/utils.py e sim/psr_sim/scenarios)
# --------------------------------------------------------------------------
V_MAX = 0.20  # m/s
W_MAX = 4.0  # rad/s
ARENA = 1.0  # meia largura da arena (m), paredes internas em ±1
PAREDE = 0.02  # espessura da parede (m)
RAIO_CILINDRO = 0.15  # m, obstáculo do avoid
RAIO_ROBO = 0.037  # m
LIMIAR_COLISAO = 0.045  # m, definição de colisão do enunciado
RAIO_META = 0.06  # m, distância para a meta contar no goto
MARGEM_METAS = 0.25  # metas sorteadas em ±(1 - 0,25)
ALINHADO = math.radians(10.0)  # critério de alinhamento usado nas métricas

# --------------------------------------------------------------------------
# Estilo (inspirado na figura de referência: quadrada, contorno completo,
# grade pontilhada, legenda com moldura, sem título dentro da imagem)
# --------------------------------------------------------------------------
COR = ["#1f6fd6", "#e8690b"]  # Config. 1 (azul), Config. 2 (laranja)
TINTA = "#111111"             # preto dos contornos, eixos e texto
TINTA2 = "#444444"            # cinza escuro (linhas de referência secundárias)
MUDO = "#8a8a8a"              # cinza (metas pendentes, região de sorteio)
GRADE = "#c9c9c9"            # cinza claro da grade pontilhada
EIXO_CIL = "#cfcfcf"          # preenchimento do cilindro
ZONA = "#eeeeee"              # faixa sombreada (zona de colisão / alinhado)


def aplicar_estilo():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 200,
        # bbox padrão (NÃO "tight"): preserva a figura quadrada ao salvar
        "savefig.bbox": "standard",
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelcolor": TINTA,
        "axes.labelsize": 12,
        "axes.edgecolor": TINTA,
        "axes.linewidth": 1.1,
        # contorno completo (as quatro bordas), como na referência
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.bottom": True,
        "axes.spines.left": True,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRADE,
        "grid.linewidth": 0.9,
        "grid.linestyle": ":",   # grade pontilhada
        "xtick.color": TINTA,
        "ytick.color": TINTA,
        "xtick.labelcolor": TINTA,
        "ytick.labelcolor": TINTA,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.edgecolor": TINTA,
        "legend.fancybox": False,
        "legend.borderpad": 0.5,
        "lines.solid_capstyle": "round",
        "axes.unicode_minus": True,
    })


class FormatoVirgula(ticker.ScalarFormatter):
    """Rótulos dos eixos com vírgula decimal."""

    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace(".", ",")


def virgula(*axes):
    for ax in axes:
        for eixo in (ax.xaxis, ax.yaxis):
            if type(eixo.get_major_formatter()) is ticker.ScalarFormatter:
                eixo.set_major_formatter(FormatoVirgula())


def num(x, casas=2):
    """Número com vírgula decimal."""
    return f"{x:.{casas}f}".replace(".", ",")


def valor_param(v):
    s = f"{v:.3f}".rstrip("0")
    return (s + "0" if s.endswith(".") else s).replace(".", ",")


def texto_params(cfg, mathtext=True):
    partes = []
    for k, v in cfg["params"].items():
        mt, puro, un = ROTULOS.get(k, (k, k, ""))
        partes.append(f"{mt if mathtext else puro} = {valor_param(v)}{un}")
    return ("  ·  " if mathtext else "; ").join(partes)


# --------------------------------------------------------------------------
# Infraestrutura de figuras quadradas
# --------------------------------------------------------------------------
def nova(lado=5.0):
    """Figura quadrada com um único eixo e layout que cabe os rótulos dentro."""
    fig, ax = plt.subplots(figsize=(lado, lado), layout="constrained")
    return fig, ax


def quadrado(ax):
    """Força a CAIXA do eixo a ser quadrada (independente da escala dos dados)."""
    ax.set_box_aspect(1)


def zero_diagonal(fig, ax):
    """Se os dois eixos começam exatamente no mesmo valor (p.ex. 0 e 0), escreve
    esse rótulo uma única vez, na diagonal do canto, em vez de repeti-lo em x e y.

    Não faz nada quando o canto do eixo não coincide com um tick de mesmo valor
    nos dois eixos (arena, eixos em π, categóricos), então é seguro chamar sempre.
    """
    fig.canvas.draw()  # garante ticks e limites finais
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    tolx = 1e-6 * abs(x1 - x0)
    toly = 1e-6 * abs(y1 - y0)
    xt = [t for t in ax.get_xticks() if x0 - tolx <= t <= x1 + tolx]
    yt = [t for t in ax.get_yticks() if y0 - toly <= t <= y1 + toly]
    if not xt or not yt:
        return
    xs, ys = min(xt), min(yt)
    # os dois eixos precisam COMEÇAR exatamente nesse tick (canto = valor)
    if abs(xs - x0) > max(tolx, 1e-9) or abs(ys - y0) > max(toly, 1e-9):
        return
    # os dois eixos precisam começar no MESMO valor numérico (ex.: 0 e 0)
    if abs(xs - ys) > 1e-6 * max(1.0, abs(xs), abs(ys)):
        return
    fx = ax.xaxis.get_major_formatter()
    fy = ax.yaxis.get_major_formatter()
    try:
        sx, sy = fx(xs), fy(ys)
    except Exception:
        return
    # rótulo único do canto: "0" na origem; senão o mais informativo dos dois
    if abs(xs) <= 1e-12:
        lx = "0"
    else:
        lx = sx if len(sx) >= len(sy) else sy
    if not lx:
        return
    # apaga o primeiro rótulo de cada eixo e escreve um único rótulo na diagonal
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, pos, f=fx, t=xs, e=tolx: "" if abs(v - t) <= max(e, 1e-9) else f(v, pos)))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, pos, f=fy, t=ys, e=toly: "" if abs(v - t) <= max(e, 1e-9) else f(v, pos)))
    ax.annotate(lx, xy=(0, 0), xycoords="axes fraction", xytext=(-11, -12),
                textcoords="offset points", ha="center", va="center",
                fontsize=plt.rcParams["xtick.labelsize"], color=TINTA,
                annotation_clip=False)


def legenda(ax, handles=None, loc="upper right", **kw):
    leg = ax.legend(handles=handles, loc=loc, **kw)
    if leg is not None:
        leg.get_frame().set_linewidth(0.9)
        leg.set_zorder(30)
    return leg


def legenda_baixo(fig, handles=None, ncol=3, **kw):
    """Legenda fora dos eixos, logo abaixo do gráfico, para não cobrir os dados.
    Usa legenda de figura com layout 'constrained', então a figura continua
    quadrada (o eixo encolhe para abrir espaço)."""
    leg = fig.legend(handles=handles, loc="outside lower center", ncol=ncol,
                     frameon=True, edgecolor=TINTA, facecolor="white",
                     framealpha=1.0, fancybox=False, borderpad=0.5,
                     handlelength=1.6, columnspacing=1.4, **kw)
    leg.get_frame().set_linewidth(0.9)
    return leg


def eixos_pi(ax, eixo="y", meio=False):
    if meio:
        vals = [0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi]
        nomes = ["0", "π/4", "π/2", "3π/4", "π"]
    else:
        vals = [-math.pi, -math.pi / 2, 0, math.pi / 2, math.pi]
        nomes = ["−π", "−π/2", "0", "π/2", "π"]
    (ax.set_yticks if eixo == "y" else ax.set_xticks)(vals, nomes)


def desenhar_arena(ax, cilindro=False, regiao_metas=False):
    a, t = ARENA, PAREDE
    paredes = [(-a - t, a, 2 * (a + t), t), (-a - t, -a - t, 2 * (a + t), t),
               (-a - t, -a, t, 2 * a), (a, -a, t, 2 * a)]
    for x0, y0, w, h in paredes:
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor=TINTA2, edgecolor="none", zorder=3))
    if cilindro:
        ax.add_patch(Circle((0, 0), RAIO_CILINDRO, facecolor=EIXO_CIL, edgecolor=MUDO, lw=0.8,
                            zorder=3))
    if regiao_metas:
        m = a - MARGEM_METAS
        ax.add_patch(Rectangle((-m, -m), 2 * m, 2 * m, fill=False, edgecolor=MUDO, lw=0.9,
                               ls=(0, (4, 3)), zorder=1))
    lim = a + t + 0.04
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def salvar(fig, saida, nome, formatos, mostrar):
    for fmt in formatos:
        caminho = saida / f"{nome}.{fmt}"
        fig.savefig(caminho)
        print(f"  {caminho}")
    if not mostrar:
        plt.close(fig)


# --------------------------------------------------------------------------
# Dados e métricas (inalterados em relação à versão anterior)
# --------------------------------------------------------------------------
def carregar(caminho, duracao):
    d = np.genfromtxt(caminho, delimiter=",", names=True)
    dados = {k: np.asarray(d[k], dtype=float) for k in d.dtype.names}
    if duracao and duracao > 0:
        manter = dados["t"] <= duracao
        dados = {k: v[manter] for k, v in dados.items()}
    return dados


def comprimento(x, y):
    ok = ~(np.isnan(x) | np.isnan(y))
    return float(np.sum(np.hypot(np.diff(x[ok]), np.diff(y[ok]))))


def metricas_avoid(d):
    t, x, y, r, v, w = (d[k] for k in ("t", "x", "y", "min_range", "v", "w"))
    passos = np.r_[0.0, np.hypot(np.diff(x), np.diff(y))]
    passos[np.isnan(passos)] = 0.0
    abaixo = r < LIMIAR_COLISAO
    inicio_colisao = np.flatnonzero(abaixo & ~np.r_[False, abaixo[:-1]])
    dt = float(np.median(np.diff(t)))

    lado = 0.05
    bordas = np.arange(-ARENA, ARENA + lado / 2, lado)
    centros = (bordas[:-1] + bordas[1:]) / 2
    cx, cy = np.meshgrid(centros, centros, indexing="ij")
    livre = ((np.abs(cx) <= ARENA - RAIO_ROBO) & (np.abs(cy) <= ARENA - RAIO_ROBO)
             & (np.hypot(cx, cy) >= RAIO_CILINDRO + RAIO_ROBO))
    ok = ~(np.isnan(x) | np.isnan(y))
    hist, _, _ = np.histogram2d(x[ok], y[ok], bins=[bordas, bordas])
    cobertura = float(np.sum((hist > 0) & livre) / np.sum(livre))

    i_min = int(np.nanargmin(r))
    return {
        "duracao": float(t[-1]),
        "distancia": float(np.sum(passos)),
        "colisoes": int(len(inicio_colisao)),
        "idx_colisoes": inicio_colisao,
        "tempo_abaixo": float(np.sum(abaixo) * dt),
        "i_min": i_min,
        "d_min": float(r[i_min]),
        "t_min": float(t[i_min]),
        "d_media": float(np.nanmean(r)),
        "d_p5": float(np.nanpercentile(r, 5)),
        "v_media": float(np.mean(v)),
        "w_abs_media": float(np.mean(np.abs(w))),
        "frac_freando": float(np.mean(v < V_MAX - 1e-6)),
        "frac_w_saturado": float(np.mean(np.abs(w) >= W_MAX - 1e-6)),
        "cobertura": cobertura,
    }


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def segmentos_goto(d):
    """Divide o ensaio em trechos, um por meta. A meta muda no log quando é atingida."""
    t, x, y, yaw, gx, gy = (d[k] for k in ("t", "x", "y", "yaw", "gx", "gy"))
    rho = np.hypot(gx - x, gy - y)
    alpha = wrap(np.arctan2(gy - y, gx - x) - yaw)
    troca = np.flatnonzero((np.diff(gx) != 0) | (np.diff(gy) != 0)) + 1
    inicios = np.r_[0, troca]
    fins = np.r_[troca, len(t)]
    segs = []
    for k, (i0, i1) in enumerate(zip(inicios, fins)):
        atingida = k < len(troca)
        tau = t[i0:i1] - t[i0]
        a = alpha[i0:i1]
        dentro = np.flatnonzero(np.abs(a) < ALINHADO)
        fim_caminho = i1 + 1 if atingida else i1
        comp = comprimento(x[i0:fim_caminho], y[i0:fim_caminho])
        rho0 = float(rho[i0])
        segs.append({
            "k": k + 1,
            "atingida": atingida,
            "meta": (float(gx[i0]), float(gy[i0])),
            "t_inicio": float(t[i0]),
            "t_chegada": float(t[i1]) if atingida else float("nan"),
            "tempo": float(t[i1] - t[i0]) if atingida else float("nan"),
            "rho0": rho0,
            "alpha0": float(alpha[i0]),
            "t_alinhar": float(tau[dentro[0]]) if len(dentro) else float("nan"),
            "comprimento": comp,
            "retidao": (rho0 - RAIO_META) / comp if comp > 0 else float("nan"),
            "tau": tau,
            "rho": rho[i0:i1],
            "alpha": a,
        })
    return rho, alpha, segs


def metricas_goto(d, segs):
    ok = [s for s in segs if s["atingida"]]
    tempos = np.array([s["tempo"] for s in ok])
    rho0 = np.array([s["rho0"] for s in ok])
    vazio = len(ok) == 0
    return {
        "duracao": float(d["t"][-1]),
        "metas": len(ok),
        "tempo_medio": float("nan") if vazio else float(np.mean(tempos)),
        "tempo_dp": float("nan") if vazio else float(np.std(tempos, ddof=1)) if len(ok) > 1 else 0.0,
        "tempo_mediana": float("nan") if vazio else float(np.median(tempos)),
        "tempo_min": float("nan") if vazio else float(np.min(tempos)),
        "tempo_max": float("nan") if vazio else float(np.max(tempos)),
        "t_alinhar": float(np.nanmean([s["t_alinhar"] for s in ok])) if ok else float("nan"),
        "rho0_medio": float("nan") if vazio else float(np.mean(rho0)),
        "vel_aprox": float("nan") if vazio else float(np.sum(rho0) / np.sum(tempos)),
        "retidao": float(np.nanmean([s["retidao"] for s in ok])) if ok else float("nan"),
        "distancia": comprimento(d["x"], d["y"]),
        "v_media": float(np.mean(d["v"])),
        "w_abs_media": float(np.mean(np.abs(d["w"]))),
    }


def preparar(cfgs, caminhos, duracao, tipo):
    runs = []
    for i, cfg in enumerate(cfgs):
        caminho = Path(caminhos[i]) if caminhos else AQUI / cfg["csv"]
        d = carregar(caminho, duracao)
        run = {**cfg, "csv": str(caminho), "cor": COR[i], "dados": d}
        if tipo == "avoid":
            run["met"] = metricas_avoid(d)
        else:
            rho, alpha, segs = segmentos_goto(d)
            run.update(rho=rho, alpha=alpha, segs=segs, met=metricas_goto(d, segs))
        runs.append(run)
    return runs


def estatistica_por_faixa(x, y, bordas, n_min=15):
    idx = np.digitize(x, bordas) - 1
    linhas = []
    for i in range(len(bordas) - 1):
        a = y[idx == i]
        if len(a) >= n_min:
            q1, med, q3 = np.percentile(a, [25, 50, 75])
            linhas.append(((bordas[i] + bordas[i + 1]) / 2, np.mean(a), med, q1, q3))
    if not linhas:
        return {k: np.array([]) for k in ("x", "media", "mediana", "q1", "q3")}
    c = np.array(linhas)
    return {"x": c[:, 0], "media": c[:, 1], "mediana": c[:, 2], "q1": c[:, 3], "q3": c[:, 4]}


def curva_alinhada(segs, chave, passo=0.05, frac_min=0.25):
    segs = [s for s in segs if s["atingida"] and len(s["tau"]) > 1]
    if not segs:
        return np.array([]), np.array([]), np.array([]), np.array([])
    grade = np.arange(0, max(s["tau"][-1] for s in segs) + passo, passo)
    M = np.full((len(segs), len(grade)), np.nan)
    for i, s in enumerate(segs):
        dentro = grade <= s["tau"][-1]
        M[i, dentro] = np.interp(grade[dentro], s["tau"], s[chave])
    ativos = np.sum(~np.isnan(M), axis=0)
    ok = ativos >= max(3, frac_min * len(segs))
    q1, med, q3 = np.nanpercentile(M[:, ok], [25, 50, 75], axis=0)
    return grade[ok], med, q1, q3


def histograma_pct(ax, valores, bordas, cor, rotulo=None):
    pesos = np.full(len(valores), 100.0 / len(valores))
    h, _ = np.histogram(valores, bins=bordas, weights=pesos)
    ax.stairs(h, bordas, color=cor, lw=1.8, zorder=3, label=rotulo)
    ax.stairs(h, bordas, fill=True, color=cor, alpha=0.12, lw=0, zorder=2)


# --------------------------------------------------------------------------
# Figuras do avoid (uma por arquivo)
# --------------------------------------------------------------------------
def fig_avoid_trajetoria(r):
    fig, ax = nova(5.2)
    d, m = r["dados"], r["met"]
    desenhar_arena(ax, cilindro=True)
    ax.plot(d["x"], d["y"], color=r["cor"], lw=0.8, alpha=0.7, zorder=2)
    ax.plot(d["x"][0], d["y"][0], "o", ms=8, mfc="white", mec=TINTA, mew=1.4, zorder=6)
    i = m["i_min"]
    ax.plot(d["x"][i], d["y"][i], "o", ms=7, color=TINTA, mec="white", mew=1.0, zorder=7)
    alcas = [
        Line2D([], [], color=r["cor"], lw=2, label="trajetória"),
        Line2D([], [], ls="", marker="o", ms=8, mfc="white", mec=TINTA, mew=1.4, label="início"),
        Line2D([], [], ls="", marker="o", ms=7, color=TINTA, mec="white",
               label=f"menor distância ({num(m['d_min'], 3)} m)"),
    ]
    if m["colisoes"]:
        idx = m["idx_colisoes"]
        ax.plot(d["x"][idx], d["y"][idx], "x", ms=8, mew=1.6, color=TINTA, zorder=8)
        alcas.append(Line2D([], [], ls="", marker="x", ms=8, mew=1.6, color=TINTA,
                            label="colisão (d < 0,045 m)"))
    legenda_baixo(fig, alcas, ncol=3)
    virgula(ax)
    return fig


def fig_avoid_dmin(r, t_max):
    fig, ax = nova()
    d, m = r["dados"], r["met"]
    ax.axhspan(0, LIMIAR_COLISAO, color=ZONA, lw=0, zorder=0)
    ax.plot(d["t"], d["min_range"], color=r["cor"], lw=1.0, zorder=3,
            label="menor distância do lidar")
    ax.axhline(LIMIAR_COLISAO, color=TINTA, lw=1.1, ls=(0, (5, 3)), zorder=4,
               label="limiar de colisão (0,045 m)")
    d_slow = r["params"].get("D_SLOW")
    if d_slow is not None:
        ax.axhline(d_slow, color=TINTA2, lw=1.1, ls=(0, (1, 2)), zorder=4,
                   label=f"$D_{{\\mathrm{{slow}}}}$ = {valor_param(d_slow)} m")
    ax.plot(m["t_min"], m["d_min"], "o", ms=6, color=TINTA, mec="white", mew=1.0, zorder=6)
    ax.set_xlabel("tempo (s)")
    ax.set_ylabel("menor distância do lidar (m)")
    ax.set_ylim(0, 0.53)
    ax.set_xlim(0, t_max)
    legenda(ax, loc="lower right")
    quadrado(ax)
    virgula(ax)
    zero_diagonal(fig, ax)
    return fig


def fig_avoid_hist_dmin(runs):
    fig, ax = nova()
    bordas = np.arange(0, 0.5 + 0.0101, 0.01)
    for r in runs:
        histograma_pct(ax, np.clip(r["dados"]["min_range"], 0, 0.5 - 1e-9), bordas, r["cor"],
                       rotulo=r["nome"])
    ax.axvline(LIMIAR_COLISAO, color=TINTA, lw=1.1, ls=(0, (5, 3)), zorder=5,
               label="limiar de colisão")
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("menor distância do lidar (m)")
    ax.set_ylabel("% do tempo")
    legenda(ax, loc="upper left")
    quadrado(ax)
    virgula(ax)
    zero_diagonal(fig, ax)
    return fig


def fig_avoid_hist_omega(runs):
    fig, ax = nova()
    passo = 0.2
    bordas = np.arange(-W_MAX - passo / 2, W_MAX + passo, passo)
    for r in runs:
        histograma_pct(ax, r["dados"]["w"], bordas, r["cor"], rotulo=r["nome"])
    ax.set_xlim(-W_MAX, W_MAX)
    ax.set_xlabel("comando angular ω (rad/s)")
    ax.set_ylabel("% do tempo")
    legenda(ax, loc="upper right")
    quadrado(ax)
    virgula(ax)
    zero_diagonal(fig, ax)
    return fig


def fig_avoid_vel_dmin(runs):
    fig, ax = nova()
    bordas = np.arange(0.06, 0.5 + 0.001, 0.02)
    for r in runs:
        e = estatistica_por_faixa(r["dados"]["min_range"], r["dados"]["v"], bordas)
        ax.plot(e["x"], e["media"], color=r["cor"], lw=2.0, marker="o", ms=5,
                mec="white", mew=0.8, zorder=3, label=r["nome"])
    ax.axhline(V_MAX, color=TINTA2, lw=1.1, ls=(0, (1, 2)), zorder=2,
               label="$V_{max}$ = 0,20 m/s")
    ax.set_xlim(0.05, 0.5)
    ax.set_ylim(0, V_MAX * 1.12)
    ax.set_xlabel("menor distância do lidar (m)")
    ax.set_ylabel("velocidade linear média (m/s)")
    legenda(ax, loc="lower right")
    quadrado(ax)
    virgula(ax)
    zero_diagonal(fig, ax)
    return fig


# --------------------------------------------------------------------------
# Figuras do goto (uma por arquivo)
# --------------------------------------------------------------------------
def fig_goto_trajetoria(r):
    fig, ax = nova(5.2)
    d, m = r["dados"], r["met"]
    desenhar_arena(ax, regiao_metas=True)
    ax.plot(d["x"], d["y"], color=r["cor"], lw=0.8, alpha=0.7, zorder=2)
    ax.plot(d["x"][0], d["y"][0], "o", ms=8, mfc="white", mec=TINTA, mew=1.4, zorder=6)
    for s in r["segs"]:
        gx, gy = s["meta"]
        if s["atingida"]:
            ax.add_patch(Circle((gx, gy), RAIO_META, fill=False, ec=TINTA, lw=0.9, zorder=5))
            ax.text(gx, gy, str(s["k"]), fontsize=6, color=TINTA, ha="center", va="center",
                    zorder=6)
        else:
            ax.add_patch(Circle((gx, gy), RAIO_META, fill=False, ec=MUDO, lw=1.0,
                                ls=(0, (2, 2)), zorder=5))
    alcas = [
        Line2D([], [], color=r["cor"], lw=2, label="trajetória"),
        Line2D([], [], ls="", marker="o", ms=8, mfc="white", mec=TINTA, mew=1.4, label="início"),
        Line2D([], [], ls="", marker="o", ms=10, mfc="none", mec=TINTA, mew=0.9,
               label="meta atingida"),
        Line2D([], [], ls="", marker="o", ms=10, mfc="none", mec=MUDO, mew=1.0,
               label="meta pendente"),
        Line2D([], [], color=MUDO, lw=0.9, ls=(0, (4, 3)), label="região de sorteio"),
    ]
    legenda_baixo(fig, alcas, ncol=3)
    virgula(ax)
    return fig


def fig_goto_rho(runs, t_max):
    fig, ax = nova()
    for r in runs:
        ax.plot(r["dados"]["t"], r["rho"], color=r["cor"], lw=0.9, zorder=3, label=r["nome"])
    ax.axhline(RAIO_META, color=TINTA, lw=1.1, ls=(0, (5, 3)), zorder=4,
               label="raio de chegada (0,06 m)")
    ax.set_xlabel("tempo (s)")
    ax.set_ylabel("ρ — distância até a meta (m)")
    ax.set_xlim(0, t_max)
    ax.set_ylim(0, None)
    legenda_baixo(fig, ncol=3)
    quadrado(ax)
    virgula(ax)
    zero_diagonal(fig, ax)
    return fig


def fig_goto_alpha(runs, t_max):
    fig, ax = nova()
    ax.axhline(0, color=MUDO, lw=0.9, zorder=1)
    for r in runs:
        ax.plot(r["dados"]["t"], r["alpha"], color=r["cor"], lw=0.8, zorder=3, label=r["nome"])
    ax.set_xlabel("tempo (s)")
    ax.set_ylabel("α — erro de orientação (rad)")
    ax.set_xlim(0, t_max)
    ax.set_ylim(-math.pi * 1.05, math.pi * 1.05)
    eixos_pi(ax)
    legenda_baixo(fig, ncol=2)
    quadrado(ax)
    virgula(ax)
    zero_diagonal(fig, ax)
    return fig


def fig_goto_vel_rho(runs):
    fig, ax = nova()
    bordas = np.arange(0.05, 0.40 + 0.001, 0.01)
    for r in runs:
        e = estatistica_por_faixa(r["rho"], r["dados"]["v"], bordas)
        ax.fill_between(e["x"], e["q1"], e["q3"], color=r["cor"], alpha=0.15, lw=0)
        ax.plot(e["x"], e["mediana"], color=r["cor"], lw=2.0, zorder=3, label=r["nome"])
        xs = np.linspace(bordas[0], bordas[-1], 200)
        ax.plot(xs, np.clip(r["params"]["K_RHO"] * xs, 0, V_MAX), color=r["cor"], lw=1.0,
                ls=(0, (1, 2)), zorder=4)
    ax.axvline(RAIO_META, color=TINTA, lw=1.1, ls=(0, (5, 3)), zorder=5,
               label="raio de chegada (0,06 m)")
    ax.set_xlim(bordas[0], bordas[-1])
    ax.set_ylim(0, V_MAX * 1.12)
    ax.set_xlabel("ρ — distância até a meta (m)")
    ax.set_ylabel("velocidade linear v (m/s)")
    handles = [Line2D([], [], color=r["cor"], lw=2.0, label=r["nome"]) for r in runs]
    handles.append(Line2D([], [], color=TINTA2, lw=1.0, ls=(0, (1, 2)),
                          label="lei v = min($K_\\rho$ρ, $V_{max}$)"))
    handles.append(Line2D([], [], color=TINTA, lw=1.1, ls=(0, (5, 3)),
                          label="raio de chegada"))
    legenda(ax, handles, loc="lower right")
    quadrado(ax)
    virgula(ax)
    zero_diagonal(fig, ax)
    return fig


def fig_goto_alpha_resp(runs, janela=2.5):
    fig, ax = nova()
    ax.axhspan(-ALINHADO, ALINHADO, color=ZONA, lw=0, zorder=0)
    ax.axhline(0, color=MUDO, lw=0.9, zorder=1)
    for r in runs:
        segs_s = [{**s, "alpha_s": s["alpha"] * (1.0 if s["alpha0"] >= 0 else -1.0)}
                  for s in r["segs"]]
        tau, med, q1, q3 = curva_alinhada(segs_s, "alpha_s")
        ax.fill_between(tau, q1, q3, color=r["cor"], alpha=0.15, lw=0)
        ax.plot(tau, med, color=r["cor"], lw=2.0, label=r["nome"])
    ax.set_xlim(0, janela)
    ax.set_ylim(-0.35, math.pi * 1.03)
    eixos_pi(ax, meio=True)
    ax.set_xlabel("tempo desde a nova meta (s)")
    ax.set_ylabel("α · sinal(α₀) (rad)")
    handles = [Line2D([], [], color=r["cor"], lw=2.0, label=r["nome"]) for r in runs]
    handles.append(Line2D([], [], color=MUDO, lw=6, alpha=0.4, label="faixa ±10° (alinhado)"))
    legenda(ax, handles, loc="upper right")
    quadrado(ax)
    virgula(ax)
    zero_diagonal(fig, ax)
    return fig


def fig_goto_tempo_rho0(runs):
    fig, ax = nova()
    for r in runs:
        ok = [s for s in r["segs"] if s["atingida"]]
        rho0 = np.array([s["rho0"] for s in ok])
        tempos = np.array([s["tempo"] for s in ok])
        ax.scatter(rho0, tempos, s=30, color=r["cor"], edgecolors="white", linewidths=0.8,
                   zorder=3, label=r["nome"])
        if len(ok) > 2:
            a, b = np.polyfit(rho0, tempos, 1)
            xs = np.array([rho0.min(), rho0.max()])
            ax.plot(xs, a * xs + b, color=r["cor"], lw=1.6, zorder=2)
    ax.set_xlabel("ρ₀ — distância inicial até a meta (m)")
    ax.set_ylabel("tempo até a meta (s)")
    ax.set_ylim(bottom=0)
    legenda(ax, loc="upper left")
    quadrado(ax)
    virgula(ax)
    zero_diagonal(fig, ax)
    return fig


def fig_goto_tempo_meta(runs):
    fig, ax = nova()
    rng = np.random.default_rng(0)
    for i, r in enumerate(runs):
        ok = [s for s in r["segs"] if s["atingida"]]
        tempos = np.array([s["tempo"] for s in ok])
        jitter = rng.uniform(-0.14, 0.14, len(tempos))
        ax.scatter(np.full(len(tempos), i) + jitter, tempos, s=26, color=r["cor"],
                   edgecolors="white", linewidths=0.8, zorder=3)
        m = r["met"]
        ax.hlines(m["tempo_medio"], i - 0.3, i + 0.3, color=TINTA, lw=2.0, zorder=4)
        ax.text(i, m["tempo_medio"] + 0.25,
                f"média {num(m['tempo_medio'], 2)} s",
                va="bottom", ha="center", fontsize=9, color=TINTA, zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.set_xticks(range(len(runs)), [r["nome"] for r in runs])
    ax.set_xlim(-0.5, len(runs) - 0.5)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("tempo até a meta (s)")
    ax.grid(axis="x", visible=False)
    handles = [Line2D([], [], ls="", marker="o", ms=7, color=r["cor"], mec="white",
                      label=r["nome"]) for r in runs]
    handles.append(Line2D([], [], color=TINTA, lw=2.0, label="média"))
    legenda(ax, handles, loc="upper right")
    quadrado(ax)
    virgula(ax)
    zero_diagonal(fig, ax)
    return fig


# --------------------------------------------------------------------------
# Resumo em texto
# --------------------------------------------------------------------------
def tabela_md(titulo_tab, runs, linhas):
    cab = "| Métrica | " + " | ".join(f"{r['nome']} ({texto_params(r, False)})" for r in runs) + " |"
    sep = "|---|" + "---:|" * len(runs)
    corpo = ["| " + nome + " | " + " | ".join(f(r["met"]) for r in runs) + " |"
             for nome, f in linhas]
    arquivos = "\n".join(f"- {r['nome']}: `{Path(r['csv']).name}`" for r in runs)
    return f"## {titulo_tab}\n\n{arquivos}\n\n" + "\n".join([cab, sep, *corpo]) + "\n"


def resumo(avoid, goto):
    partes = ["# Resumo dos ensaios\n"]
    if avoid:
        partes.append(tabela_md("Desvio de obstáculos (avoid)", avoid, [
            ("Duração analisada (s)", lambda m: num(m["duracao"], 1)),
            ("Distância percorrida (m)", lambda m: num(m["distancia"], 2)),
            ("Colisões (menor distância < 0,045 m)", lambda m: str(m["colisoes"])),
            ("Tempo abaixo do limiar de colisão (s)", lambda m: num(m["tempo_abaixo"], 2)),
            ("Menor distância registrada (m)", lambda m: num(m["d_min"], 3)),
            ("Menor distância média (m)", lambda m: num(m["d_media"], 3)),
            ("5º percentil da menor distância (m)", lambda m: num(m["d_p5"], 3)),
            ("Velocidade linear média (m/s)", lambda m: num(m["v_media"], 3)),
            ("Tempo com v < V_MAX (%)", lambda m: num(100 * m["frac_freando"], 1)),
            ("\\|ω\\| médio (rad/s)", lambda m: num(m["w_abs_media"], 3)),
            ("Tempo com \\|ω\\| = W_MAX (%)", lambda m: num(100 * m["frac_w_saturado"], 1)),
            ("Cobertura da arena (células de 5 cm, %)", lambda m: num(100 * m["cobertura"], 1)),
        ]))
    if goto:
        partes.append(tabela_md("Navegação até metas (goto)", goto, [
            ("Duração analisada (s)", lambda m: num(m["duracao"], 1)),
            ("Metas atingidas", lambda m: str(m["metas"])),
            ("Tempo médio por meta (s)", lambda m: f"{num(m['tempo_medio'], 2)} ± {num(m['tempo_dp'], 2)}"),
            ("Mediana do tempo por meta (s)", lambda m: num(m["tempo_mediana"], 2)),
            ("Tempo mínimo – máximo (s)", lambda m: f"{num(m['tempo_min'], 2)} – {num(m['tempo_max'], 2)}"),
            ("Tempo médio para alinhar (\\|α\\| < 10°) (s)", lambda m: num(m["t_alinhar"], 2)),
            ("Distância inicial média até a meta (m)", lambda m: num(m["rho0_medio"], 3)),
            ("Velocidade média de aproximação Σρ₀/Σt (m/s)", lambda m: num(m["vel_aprox"], 3)),
            ("Retidão média do caminho (ρ₀ − 0,06)/percorrido", lambda m: num(m["retidao"], 3)),
            ("Distância total percorrida (m)", lambda m: num(m["distancia"], 2)),
            ("Velocidade linear média (m/s)", lambda m: num(m["v_media"], 3)),
            ("\\|ω\\| médio (rad/s)", lambda m: num(m["w_abs_media"], 3)),
        ]))
    return "\n".join(partes)


def salvar_metas_csv(goto, caminho):
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "meta", "gx", "gy", "t_inicio", "t_chegada", "tempo", "rho0",
                    "alpha0", "t_alinhar", "comprimento", "retidao"])
        for r in goto:
            for s in r["segs"]:
                if s["atingida"]:
                    w.writerow([r["nome"], s["k"], *(round(v, 4) for v in s["meta"]),
                                *(round(s[k], 4) for k in ("t_inicio", "t_chegada", "tempo", "rho0",
                                                           "alpha0", "t_alinhar", "comprimento",
                                                           "retidao"))])


# --------------------------------------------------------------------------
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--saida", default=str(AQUI / "graficos"), help="pasta de saída")
    p.add_argument("--formatos", nargs="+", default=["png", "pdf"], help="ex.: png pdf svg")
    p.add_argument("--duracao-avoid", type=float, default=300.0,
                   help="corta o log do avoid nesse tempo (s); 0 = sem corte")
    p.add_argument("--duracao-goto", type=float, default=180.0,
                   help="corta o log do goto nesse tempo (s); 0 = sem corte")
    p.add_argument("--avoid", nargs=2, metavar="CSV", help="CSVs do avoid (config. 1 e 2)")
    p.add_argument("--goto", nargs=2, metavar="CSV", help="CSVs do goto (config. 1 e 2)")
    p.add_argument("--mostrar", action="store_true", help="abre as figuras na tela")
    args = p.parse_args()

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    aplicar_estilo()

    avoid = preparar(CONFIG_AVOID, args.avoid, args.duracao_avoid, "avoid")
    goto = preparar(CONFIG_GOTO, args.goto, args.duracao_goto, "goto")

    t_avoid = max(r["met"]["duracao"] for r in avoid)
    t_goto = max(r["met"]["duracao"] for r in goto)

    print("Figuras:")
    # avoid
    salvar(fig_avoid_trajetoria(avoid[0]), saida, "avoid_trajetoria_cfg1", args.formatos, args.mostrar)
    salvar(fig_avoid_trajetoria(avoid[1]), saida, "avoid_trajetoria_cfg2", args.formatos, args.mostrar)
    salvar(fig_avoid_dmin(avoid[0], t_avoid), saida, "avoid_dmin_cfg1", args.formatos, args.mostrar)
    salvar(fig_avoid_dmin(avoid[1], t_avoid), saida, "avoid_dmin_cfg2", args.formatos, args.mostrar)
    salvar(fig_avoid_hist_dmin(avoid), saida, "avoid_hist_dmin", args.formatos, args.mostrar)
    salvar(fig_avoid_hist_omega(avoid), saida, "avoid_hist_omega", args.formatos, args.mostrar)
    salvar(fig_avoid_vel_dmin(avoid), saida, "avoid_vel_dmin", args.formatos, args.mostrar)
    # goto
    salvar(fig_goto_trajetoria(goto[0]), saida, "goto_trajetoria_cfg1", args.formatos, args.mostrar)
    salvar(fig_goto_trajetoria(goto[1]), saida, "goto_trajetoria_cfg2", args.formatos, args.mostrar)
    salvar(fig_goto_rho(goto, t_goto), saida, "goto_rho", args.formatos, args.mostrar)
    salvar(fig_goto_alpha(goto, t_goto), saida, "goto_alpha", args.formatos, args.mostrar)
    salvar(fig_goto_vel_rho(goto), saida, "goto_vel_rho", args.formatos, args.mostrar)
    salvar(fig_goto_alpha_resp(goto), saida, "goto_alpha_resp", args.formatos, args.mostrar)
    salvar(fig_goto_tempo_rho0(goto), saida, "goto_tempo_rho0", args.formatos, args.mostrar)
    salvar(fig_goto_tempo_meta(goto), saida, "goto_tempo_meta", args.formatos, args.mostrar)

    texto = resumo(avoid, goto)
    (saida / "resumo.md").write_text(texto, encoding="utf-8")
    salvar_metas_csv(goto, saida / "goto_metas.csv")
    print(f"  {saida / 'resumo.md'}\n  {saida / 'goto_metas.csv'}\n")
    print(texto)

    if args.mostrar:
        plt.show()


if __name__ == "__main__":
    main()
