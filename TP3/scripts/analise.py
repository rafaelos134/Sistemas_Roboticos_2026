#!/usr/bin/env python3
"""Análise do TP3: lê os CSVs do coleta.py, ajusta os modelos, calcula o
alcance e gera as figuras do relatório (todas com distância em mm).

    python3 scripts/analise.py dados --figs figuras
    python3 scripts/analise.py dados --header firmware/distancia/include/calibracao.h

As regras são as mesmas da planilha TP3_planilha.ods (Seção 4 do relatório):
  - saturação: o fototransistor saturado não vai a 0, para num patamar
    (V_CE(sat) ≈ 0,2-0,4 V, ~40-80 contagens, que muda com o resistor). O
    limiar é medido: menor adc_on médio da ida + --sat-margem, desde que esse
    mínimo seja menor que --sat-max; senão, considera que não houve saturação;
  - pico: maior S entre os pontos não saturados da ida;
  - limiar: S_lim = mu0 + k*sigma0, com o fundo no tempo da caracterização;
  - alcance de detecção: maior d contínua a partir do pico com S >= S_lim;
  - alcance de medição: idem, com sigma_d <= --res * d;
  - ajuste: pontos da ida entre o pico e o alcance de detecção, não saturados;
  - validação: pontos da volta dentro da faixa de ajuste.
"""

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Paleta categórica fixa (mesma ordem em todas as figuras) e marcadores
# distintos, para que as séries não dependam só da cor (impressão P&B).
CORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARCAS = ["o", "s", "^", "D"]
CINZA = "#808080"

plt.rcParams.update({
    "figure.figsize": (6.0, 3.8),
    "figure.dpi": 150,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.6,
    "lines.linewidth": 2,
    "lines.markersize": 5,
    "legend.frameon": False,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Leitura
# ---------------------------------------------------------------------------
def le_dados(pasta):
    ciclos, degraus, vivo = [], {}, []
    arquivos = sorted(glob.glob(os.path.join(pasta, "*.csv")))
    if not arquivos:
        sys.exit(f"nenhum CSV em {pasta}")
    for arq in arquivos:
        with open(arq, newline="") as f:
            leitor = csv.DictReader(f)
            campos = leitor.fieldnames or []
            linhas = list(leitor)
        if "serie" in campos:
            for l in linhas:
                ciclos.append((l["serie"], int(l["t_on_us"]), int(l["t_off_us"]),
                               float(l["d_mm"]) if l["d_mm"] else None,
                               int(l["adc_on"]), int(l["adc_off"]), int(l["sinal"])))
        elif "a0_liga" in campos:
            nome = os.path.splitext(os.path.basename(arq))[0]
            degraus[nome] = {k: np.array([float(l[k]) for l in linhas]) for k in campos}
        elif "d_regua_mm" in campos:
            vivo += [(float(l["d_regua_mm"]), float(l["mm"])) for l in linhas]
        else:
            print(f"aviso: {arq} ignorado (colunas desconhecidas)")
    print(f"{len(ciclos)} ciclos, {len(degraus)} degrau(s), {len(vivo)} leituras ao vivo")
    return ciclos, degraus, vivo


def agrupa(ciclos):
    """(serie, t_on, t_off, d) -> estatísticas."""
    g = defaultdict(list)
    for serie, ton, toff, d, on, off, s in ciclos:
        g[(serie, ton, toff, d)].append((on, off, s))
    est = {}
    for k, v in g.items():
        a = np.array(v, dtype=float)
        n = len(a)
        est[k] = {"n": n, "on": a[:, 0].mean(), "off": a[:, 1].mean(), "S": a[:, 2].mean(),
                  "sd": a[:, 2].std(ddof=1) if n > 1 else np.nan}
    return est


def curva(est, serie, ton, toff=None):
    """Arrays ordenados por d de uma série num tempo (toff=None: igual a ton)."""
    toff = ton if toff is None else toff
    pts = sorted((k[3], v) for k, v in est.items()
                 if k[0] == serie and k[1] == ton and k[2] == toff and k[3] is not None)
    if not pts:
        return None
    d = np.array([p[0] for p in pts])
    return {"d": d, **{c: np.array([p[1][c] for p in pts]) for c in ("n", "on", "off", "S", "sd")}}


def fundo(est, ton, toff=None):
    toff = ton if toff is None else toff
    return est.get(("fundo", ton, toff, None))


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
def alcance(d, S, ok, d_pico, limite_ok):
    """Maior d contínua a partir do pico em que limite_ok vale."""
    idx = np.where((d >= d_pico) & ok)[0]
    melhor = None
    for i in idx:
        if limite_ok[i]:
            melhor = d[i]
        else:
            break
    return melhor


def estima(modelo, S, p):
    S = np.asarray(S, dtype=float)
    if modelo == "M1":
        return p["g0"] + p["g1"] / np.sqrt(S)
    if modelo == "M2":
        return (p["alfa"] / S) ** (1 / p["n"])
    # M3: interpolação linear na tabela (S crescente para np.interp)
    ordem = np.argsort(p["tab_S"])
    out = np.interp(S, p["tab_S"][ordem], p["tab_d"][ordem], left=np.nan, right=np.nan)
    return out


def t_10_90(t, a):
    """Tempo entre 10 % e 90 % da excursão (subida ou descida)."""
    v0 = a[0]
    v1 = a[t >= 0.9 * t.max()].mean()
    l10, l90 = v0 + 0.1 * (v1 - v0), v0 + 0.9 * (v1 - v0)
    cruza = (lambda x, l: x <= l) if v1 < v0 else (lambda x, l: x >= l)
    c10, c90 = cruza(a, l10), cruza(a, l90)
    if not (c10.any() and c90.any()):
        return np.nan
    return t[np.argmax(c90)] - t[np.argmax(c10)]


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def salva(fig, pasta, nome, fmt):
    caminho = os.path.join(pasta, f"{nome}.{fmt}")
    fig.savefig(caminho)
    plt.close(fig)
    print(f"  figura: {caminho}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dados", nargs="?", default="dados", help="pasta com os CSVs (padrão: dados)")
    ap.add_argument("--figs", default="figuras", help="pasta das figuras (padrão: figuras)")
    ap.add_argument("--fmt", default="pdf", choices=["pdf", "png"], help="formato das figuras")
    ap.add_argument("--resumo", default="resultados.txt", help="arquivo de resumo dos números")
    ap.add_argument("--t-cal-ms", type=float, default=1000, help="tempo da caracterização (ms)")
    ap.add_argument("--k", type=float, default=3, help="fator do limiar de detecção")
    ap.add_argument("--sat-margem", type=float, default=5, help="margem acima do patamar de saturação (contagens)")
    ap.add_argument("--sat-max", type=float, default=200, help="patamar acima disso = não houve saturação")
    ap.add_argument("--sat", type=float, help="força um limiar fixo de saturação (adc_on <= sat)")
    ap.add_argument("--res", type=float, default=0.05, help="sigma_d máximo como fração de d")
    ap.add_argument("--descartar", default="", help="distâncias da ida fora do ajuste, ex.: 14,16")
    ap.add_argument("--header", help="escreve calibracao.h para o firmware de distância")
    ap.add_argument("--modelo", type=int, default=1, choices=[1, 2, 3], help="modelo usado no header")
    args = ap.parse_args()

    os.makedirs(args.figs, exist_ok=True)
    ciclos, degraus, vivo = le_dados(args.dados)
    est = agrupa(ciclos)
    tcal = int(round(args.t_cal_ms * 1000))
    R = []  # linhas do resumo

    def diz(txt=""):
        print(txt)
        R.append(txt)

    # ---------------- caracterização (ida) ----------------
    ida = curva(est, "ida", tcal)
    if ida is None:
        sys.exit(f"sem série 'ida' com t_on = t_off = {args.t_cal_ms} ms")
    f0 = fundo(est, tcal)
    if f0 is None:
        sys.exit("sem série 'fundo' no tempo da caracterização: S_lim não pode ser calculado")
    mu0, sd0 = f0["S"], f0["sd"]
    S_lim = mu0 + args.k * sd0
    d, S, sd, on = ida["d"], ida["S"], ida["sd"], ida["on"]
    # Limiar de saturação medido nos dados (não depende do valor exato dos resistores).
    if args.sat is not None:
        sat_lim = args.sat
    elif on.min() < args.sat_max:
        sat_lim = on.min() + args.sat_margem
    else:
        sat_lim = -1.0  # o sensor não saturou em nenhuma distância
    nao_sat = on > sat_lim
    d_pico = d[nao_sat][np.argmax(S[nao_sat])]
    d_det = alcance(d, S, np.ones_like(d, bool), d_pico, S >= S_lim)
    ultimo_acima = d[-1] if S[-1] >= S_lim else None

    descartar = set(float(x) for x in args.descartar.split(",") if x.strip())
    usa = nao_sat & (d >= d_pico) & (d <= d_det) & np.array([x not in descartar for x in d])
    if usa.sum() < 3:
        sys.exit(f"só {usa.sum()} pontos no ajuste; conferir os dados da ida e o fundo")

    # M1
    z = 1 / np.sqrt(np.where(S > 0, S, np.nan))
    g1, g0 = np.polyfit(z[usa], d[usa], 1)
    r2_m1 = np.corrcoef(z[usa], d[usa])[0, 1] ** 2
    # M2
    incl, inter = np.polyfit(np.log(d[usa]), np.log(S[usa]), 1)
    n_exp, alfa = -incl, np.exp(inter)
    r2_m2 = np.corrcoef(np.log(d[usa]), np.log(S[usa]))[0, 1] ** 2
    par = {"g0": g0, "g1": g1, "n": n_exp, "alfa": alfa, "tab_S": S[usa], "tab_d": d[usa]}
    fit_min, fit_max = d[usa].min(), d[usa].max()

    sigma_d = g1 / 2 * np.where(S > 0, S, np.nan) ** -1.5 * sd
    d_med = alcance(d, S, np.ones_like(d, bool), d_pico, (sigma_d <= args.res * d) & (S >= S_lim))

    diz("== Caracterização (ida, t = %g ms) ==" % args.t_cal_ms)
    diz(f"pontos: {len(d)}, de {d.min():g} a {d.max():g} mm; N médio por ponto = {ida['n'].mean():.0f}")
    diz(f"fundo: N0 = {f0['n']}, mu0 = {mu0:.2f}, sigma0 = {sd0:.2f}, S_lim = {S_lim:.2f} contagens")
    escuro = f0["off"]
    diz(f"nível com LED apagado e sem alvo: {escuro:.0f} contagens (ideal ~1023)" + (
        "  -> abaixo do esperado: conferir R_C e a luz ambiente. O sinal é a diferença"
        " entre as leituras, então as contas continuam válidas; só a faixa útil diminui." if escuro < 950 else ""))
    diz(f"menor adc_on médio na ida: {on.min():.0f} contagens -> " + (
        f"saturação detectada, limiar adc_on <= {sat_lim:.0f}" if sat_lim > 0 else
        "sem saturação em nenhuma distância"))
    diz(f"pico (fora da saturação): {d_pico:g} mm; saturados: {', '.join(f'{x:g}' for x in d[~nao_sat]) or 'nenhum'}")
    diz(f"alcance de detecção d_det = {d_det:g} mm" + (
        "  (ATENÇÃO: o último ponto ainda está acima do limiar; medir mais longe)" if ultimo_acima else ""))
    diz(f"alcance de medição d_med = {d_med:g} mm (sigma_d <= {args.res:.0%} de d)" if d_med else
        "alcance de medição: nenhum ponto atende sigma_d <= res*d")
    diz(f"200 mm dentro do alcance de detecção? {'sim' if d_det >= 200 else 'não'}")
    diz(f"faixa de ajuste: {fit_min:g} a {fit_max:g} mm, {usa.sum()} pontos")
    diz(f"M1: g0 = {g0:.3f} mm, g1 = {g1:.3f} mm·cont^0.5, R² = {r2_m1:.4f}")
    diz(f"M2: n = {n_exp:.3f}, alfa = {alfa:.4g}, R²(log) = {r2_m2:.4f}")
    diz(f"M3: tabela com {usa.sum()} pares (S, d)")

    # ---------------- validação (volta) ----------------
    volta = curva(est, "volta", tcal)
    val = {}
    if volta is not None:
        vok = (volta["on"] > sat_lim) & (volta["S"] >= S_lim) & (volta["d"] >= fit_min) & (volta["d"] <= fit_max)
        diz()
        diz(f"== Validação (volta, {vok.sum()} pontos na faixa) ==")
        diz(f"{'modelo':8s}{'N':>4s}{'RMS':>9s}{'máx':>9s}{'viés':>9s}   (mm)")
        for m in ("M1", "M2", "M3"):
            est_d = estima(m, volta["S"][vok], par)
            e = est_d - volta["d"][vok]
            e = e[np.isfinite(e)]
            val[m] = (volta["d"][vok], est_d)
            if len(e):
                diz(f"{m:8s}{len(e):4d}{np.sqrt(np.mean(e**2)):9.2f}{np.abs(e).max():9.2f}{e.mean():9.2f}")
        dif = np.interp(ida["d"], volta["d"], volta["S"]) - ida["S"]
        comum = np.isin(ida["d"], volta["d"])
        if comum.any():
            diz(f"maior diferença ida - volta no mesmo d: {np.abs(dif[comum]).max():.1f} contagens")
    else:
        diz("\n(sem série 'volta': validação não calculada)")

    # ---------------- figuras da caracterização ----------------
    fig, ax = plt.subplots()
    ax.axhspan(0, S_lim, color=CINZA, alpha=0.15, lw=0, label=f"fundo + {args.k:g}σ")
    for i, (nome, c) in enumerate((("ida", ida), ("volta", volta))):
        if c is None:
            continue
        cheio = (c["on"] > sat_lim) & (c["S"] >= S_lim) & (c["d"] >= d_pico)
        ax.errorbar(c["d"][cheio], c["S"][cheio], yerr=c["sd"][cheio], fmt=MARCAS[i], color=CORES[i],
                    ms=4, capsize=2, lw=1, label=nome)
        ax.plot(c["d"][~cheio], c["S"][~cheio], MARCAS[i], mfc="none", color=CORES[i], ms=4,
                label=f"{nome}: fora do ajuste")
    ax.axvline(200, color=CINZA, ls=":", lw=1)
    ax.axvline(d_det, color=CINZA, ls="--", lw=1)
    ax.annotate(f"d_det = {d_det:g} mm", (d_det, ax.get_ylim()[1] * 0.9), xytext=(4, 0),
                textcoords="offset points", fontsize=8, color="#444444")
    ax.set_xlabel("distância d (mm)")
    ax.set_ylabel("sinal S (contagens)")
    ax.legend(fontsize=8)
    salva(fig, args.figs, "sinal_dist", args.fmt)

    # Mesma curva em escala log: mostra onde o sinal encontra o fundo.
    fig, ax = plt.subplots()
    ax.axhspan(0.1, S_lim, color=CINZA, alpha=0.15, lw=0, label=f"fundo + {args.k:g}σ")
    for i, (nome, c) in enumerate((("ida", ida), ("volta", volta))):
        if c is not None:
            ax.semilogy(c["d"], np.clip(c["S"], 0.1, None), "-" + MARCAS[i], color=CORES[i], lw=1.2, ms=4, label=nome)
    ax.axvline(200, color=CINZA, ls=":", lw=1)
    ax.axvline(d_det, color=CINZA, ls="--", lw=1)
    ax.set_ylim(bottom=0.5)
    ax.set_xlabel("distância d (mm)")
    ax.set_ylabel("sinal S (contagens, log)")
    ax.legend(fontsize=8)
    salva(fig, args.figs, "sinal_dist_log", args.fmt)

    fig, ax = plt.subplots()
    ax.plot(d, ida["on"], "-" + MARCAS[0], color=CORES[0], label="adc_on (LED aceso)")
    ax.plot(d, ida["off"], "-" + MARCAS[1], color=CORES[1], label="adc_off (LED apagado)")
    if sat_lim > 0:
        ax.axhline(sat_lim, color=CINZA, ls=":", lw=1, label="limiar de saturação")
    ax.set_xlabel("distância d (mm)")
    ax.set_ylabel("leitura do ADC (contagens)")
    ax.set_ylim(-20, 1043)
    ax.legend(fontsize=8)
    salva(fig, args.figs, "adc_on_off", args.fmt)

    fig, ax = plt.subplots()
    ax.plot(d, sd, "-" + MARCAS[0], color=CORES[0], label="σS na ida")
    ax.axhline(sd0, color=CINZA, ls="--", lw=1, label="σ0 (sem alvo)")
    ax.set_xlabel("distância d (mm)")
    ax.set_ylabel("desvio do sinal σS (contagens)")
    ax.legend(fontsize=8)
    salva(fig, args.figs, "ruido", args.fmt)

    fig, ax = plt.subplots()
    lim = [0, fit_max * 1.05]
    ax.plot(lim, lim, color=CINZA, ls="--", lw=1, label="y = x")
    if val:
        for i, m in enumerate(("M1", "M2", "M3")):
            x, y = val[m]
            ax.plot(x, y, MARCAS[i], color=CORES[i], label=f"{m} (validação)")
    else:
        ax.plot(d[usa], estima("M1", S[usa], par), MARCAS[0], color=CORES[0], label="M1 (calibração)")
    ax.set_xlabel("distância da régua (mm)")
    ax.set_ylabel("distância estimada (mm)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8)
    salva(fig, args.figs, "dest_dist", args.fmt)

    if val:
        fig, ax = plt.subplots()
        ax.axhline(0, color=CINZA, lw=1)
        for i, m in enumerate(("M1", "M2", "M3")):
            x, y = val[m]
            ax.plot(x, y - x, "-" + MARCAS[i], color=CORES[i], label=m, lw=1.2)
        ax.set_xlabel("distância da régua (mm)")
        ax.set_ylabel("erro d_est − d (mm)")
        ax.legend(fontsize=8)
        salva(fig, args.figs, "erro", args.fmt)

    fig, ax = plt.subplots()
    ok = np.isfinite(sigma_d) & (d >= d_pico) & nao_sat
    ax.semilogy(d[ok], sigma_d[ok], "-" + MARCAS[0], color=CORES[0], label="σd (Eq. de propagação)")
    ax.semilogy(d[ok], args.res * d[ok], color=CINZA, ls="--", lw=1, label=f"{args.res:.0%} de d")
    if d_med:
        ax.axvline(d_med, color=CINZA, ls=":", lw=1)
    ax.set_xlabel("distância d (mm)")
    ax.set_ylabel("resolução σd (mm)")
    ax.legend(fontsize=8)
    salva(fig, args.figs, "resolucao", args.fmt)

    # ---------------- ao vivo ----------------
    if vivo:
        v = defaultdict(list)
        for dr, mm in vivo:
            v[dr].append(mm)
        xs = np.array(sorted(v))
        med = np.array([np.mean(v[x]) for x in xs])
        dp = np.array([np.std(v[x], ddof=1) if len(v[x]) > 1 else 0 for x in xs])
        e = med - xs
        diz()
        diz(f"== Arduino ao vivo ({len(xs)} posições) ==")
        for x, m_, s_ in zip(xs, med, dp):
            diz(f"  régua {x:6g} mm -> {m_:7.1f} ± {s_:4.1f} mm (erro {m_ - x:+.1f})")
        diz(f"RMS = {np.sqrt(np.mean(e**2)):.2f} mm, máx = {np.abs(e).max():.2f} mm")
        fig, ax = plt.subplots()
        lim = [0, xs.max() * 1.1]
        ax.plot(lim, lim, color=CINZA, ls="--", lw=1, label="y = x")
        ax.errorbar(xs, med, yerr=dp, fmt=MARCAS[0], color=CORES[0], capsize=2, lw=1, label="Arduino")
        ax.set_xlabel("distância da régua (mm)")
        ax.set_ylabel("distância impressa pelo Arduino (mm)")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=8)
        salva(fig, args.figs, "ao_vivo", args.fmt)

    # ---------------- degrau ----------------
    if degraus:
        diz()
        diz("== Resposta ao degrau (10-90 %) ==")
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
        for i, (nome, c) in enumerate(sorted(degraus.items())):
            tr = t_10_90(c["t_us"], c["a0_liga"])
            tf = t_10_90(c["t_us_apaga"], c["a0_apaga"])
            diz(f"  {nome}: LED liga -> {tr:.0f} us; LED apaga -> {tf:.0f} us")
            cor, mk = CORES[i % 4], MARCAS[i % 4]
            axs[0].plot(c["t_us"], c["a0_liga"], color=cor, lw=1.5, label=nome, marker=mk, markevery=20, ms=4)
            axs[1].plot(c["t_us_apaga"], c["a0_apaga"], color=cor, lw=1.5, label=nome, marker=mk, markevery=20, ms=4)
        axs[0].set_title("LED acende", fontsize=10)
        axs[1].set_title("LED apaga", fontsize=10)
        for a in axs:
            a.set_xlabel("tempo após a borda (µs)")
        axs[0].set_ylabel("A0 (contagens)")
        axs[1].legend(fontsize=8)
        salva(fig, args.figs, "degrau", args.fmt)

    # ---------------- T1 ----------------
    def S_em(serie, ton, toff, dd):
        e = est.get((serie, ton, toff, dd))
        if e is None and ton == tcal and toff == tcal:
            e = est.get(("ida", ton, toff, dd))
        return e

    t1 = sorted({(k[1], k[3]) for k in est if k[0] == "T1"})
    if t1:
        tempos = sorted({t for t, _ in t1} | {tcal}, reverse=True)
        dists = sorted({dd for _, dd in t1})
        diz()
        diz("== T1: t_on = t_off = t ==")
        diz("t (ms)  " + "".join(f"S({x:g} mm)".rjust(12) for x in dists) + "     sigma0")
        rel = {}
        for t in tempos:
            linha = f"{t / 1000:<8g}"
            for x in dists:
                e = S_em("T1", t, t, x)
                ref = S_em("T1", tcal, tcal, x)
                linha += (f"{e['S']:8.1f}" + (f" {e['S'] / ref['S']:4.0%}" if ref else "     ")) if e else " " * 13
                if e and ref:
                    rel.setdefault(x, []).append((t / 1000, e["S"] / ref["S"]))
            f = fundo(est, t)
            linha += f"{f['sd']:9.2f}" if f else ""
            diz(linha)
        fig, ax = plt.subplots()
        for i, x in enumerate(dists):
            if x in rel:
                tt, rr = zip(*sorted(rel[x]))
                ax.semilogx(tt, rr, "-" + MARCAS[i % 4], color=CORES[i % 4], label=f"d = {x:g} mm")
        ax.axhline(1, color=CINZA, lw=1, ls="--")
        ax.set_xlabel("t_on = t_off (ms)")
        ax.set_ylabel("S / S(1 s)")
        ax.legend(fontsize=8)
        salva(fig, args.figs, "sinal_tempo", args.fmt)

    # ---------------- T2 ----------------
    t2 = {s: sorted((k[1], k[2], v["S"], v["sd"]) for k, v in est.items() if k[0] == s) for s in ("T2on", "T2off")}
    if t2["T2on"] or t2["T2off"]:
        diz()
        diz("== T2 ==")
        fig, ax = plt.subplots()
        for i, (s, idx, rot) in enumerate((("T2on", 0, "varia t_on"), ("T2off", 1, "varia t_off"))):
            if not t2[s]:
                continue
            t = np.array([p[idx] for p in t2[s]]) / 1000
            Sv = np.array([p[2] for p in t2[s]])
            fixo = t2[s][0][1 - idx] / 1000
            diz(f"{s} ({rot}, outro fixo em {fixo:g} ms): " +
                ", ".join(f"{a:g} ms -> {b:.1f}" for a, b in zip(t, Sv)))
            ax.semilogx(t, Sv / Sv.max(), "-" + MARCAS[i], color=CORES[i], label=f"{rot} (outro = {fixo:g} ms)")
        ax.set_xlabel("tempo variado (ms)")
        ax.set_ylabel("S / S máx")
        ax.legend(fontsize=8)
        salva(fig, args.figs, "ton_toff", args.fmt)

    # ---------------- T3 ----------------
    t3_tempos = sorted({k[1] for k in est if k[0] == "T3"} | {tcal}, reverse=True)
    if any(k[0] == "T3" for k in est):
        diz()
        diz("== T3: alcance por tempo ==")
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [2, 1]})
        alc = []
        for i, t in enumerate(t3_tempos):
            c = curva(est, "T3", t) or (curva(est, "ida", t) if t == tcal else None)
            f = fundo(est, t)
            if c is None or f is None:
                diz(f"  t = {t / 1000:g} ms: falta {'a varredura' if c is None else 'o fundo'}")
                continue
            lim_t = f["S"] + args.k * f["sd"]
            dd = alcance(c["d"], c["S"], np.ones_like(c["d"], bool), c["d"].min(), c["S"] >= lim_t)
            aviso = "  (último ponto acima do limiar: medir mais longe)" if c["S"][-1] >= lim_t else ""
            diz(f"  t = {t / 1000:g} ms: S_lim = {lim_t:.2f}, d_det = {dd} mm, ~{1000 / (2 * t / 1000):.1f} leituras/s{aviso}")
            ax.semilogy(c["d"], np.clip(c["S"], 0.1, None), "-" + MARCAS[i % 4], color=CORES[i % 4],
                        label=f"t = {t / 1000:g} ms", lw=1.2, ms=4)
            ax.axhline(lim_t, color=CORES[i % 4], ls=":", lw=1)
            if dd is not None:
                alc.append((t / 1000, dd))
        ax.set_xlabel("distância d (mm)")
        ax.set_ylabel("S (contagens, log)")
        ax.legend(fontsize=8)
        if alc:
            tt, dd = zip(*sorted(alc))
            ax2.semilogx(tt, dd, "-o", color=CORES[0])
        ax2.set_xlabel("t_on = t_off (ms)")
        ax2.set_ylabel("d_det (mm)")
        salva(fig, args.figs, "alcance_tempo", args.fmt)

    # ---------------- saídas ----------------
    with open(args.resumo, "w") as f:
        f.write("\n".join(R) + "\n")
    print(f"\nresumo em {args.resumo}")

    if args.header:
        ordem = np.argsort(-S[usa])
        tS = ", ".join(f"{x:.1f}" for x in S[usa][ordem])
        tD = ", ".join(f"{x:.1f}" for x in d[usa][ordem])
        with open(args.header, "w") as f:
            f.write(f"""// Gerado por scripts/analise.py a partir de {args.dados}/ (t = {args.t_cal_ms:g} ms).
#pragma once

// Modelo usado na conversão: 1 = roteiro, 2 = lei de potência, 3 = tabela.
#define MODELO {args.modelo}

const unsigned long T_ON_US = {tcal}UL;
const unsigned long T_OFF_US = {tcal}UL;
const int M_CONV = 8;

// M1: d = G0 + G1 / sqrt(S)   (R² = {r2_m1:.4f})
const float G0 = {g0:.4f};
const float G1 = {g1:.4f};

// M2: S = ALFA * d^(-N_EXP)   (R² log = {r2_m2:.4f})
const float ALFA = {alfa:.6g};
const float N_EXP = {n_exp:.4f};

// M3: pares (S, d) da faixa de ajuste, com S decrescente.
const int TAB_N = {usa.sum()};
const float TAB_S[TAB_N] = {{{tS}}};
const float TAB_D[TAB_N] = {{{tD}}};

const float S_LIM = {S_lim:.2f};   // mu0 + {args.k:g}*sigma0
const float D_DET = {d_det:.1f};   // mm, alcance de detecção
const float D_MIN = {fit_min:.1f};   // mm, início da faixa de ajuste
const int ADC_SAT = {int(round(sat_lim))};   // adc_on <= isto: saturado (-1 = nunca)
""")
        print(f"header em {args.header} (modelo M{args.modelo})")


if __name__ == "__main__":
    main()
