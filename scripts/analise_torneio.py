#!/usr/bin/env python3
"""Tabelas e figuras do torneio para o relatório final (seção "Partidas do torneio").

Lê as gravações ao vivo do site do torneio (ultimas_tentativas/gravacoes_ao_vivo: partidas.json com
o resultado, os eventos do juiz e o commit de cada lado; <jogo>/<id>.jsonl.gz com as amostras a ~9 Hz)
e, para as partidas do R2D2 (g5) que têm gravação, escreve:

  <tabelas>/torneio_sumo.tex    adversário & partidas & partidas vencidas & rodadas V & E & D
  <tabelas>/torneio_ctf.tex     adversário & partidas & pontos nossos & deles & pegadas nossas & deles
                                & toques nossos & deles  (toque nosso = um adversário que seguramos)
  <tabelas>/torneio_soccer.tex  adversário & partidas & gols nossos & deles
  <figuras>/torneio_{sumo,ctf,soccer}.pdf

Uso:
  uv run --with matplotlib --with numpy scripts/analise_torneio.py \
      [--src ultimas_tentativas/gravacoes_ao_vivo] [--tabelas overleaf/tabelas] [--figuras figuras] \
      [--ate "2026-11-25 18:30"]      # só partidas iniciadas até esse horário
      [--commits 09e24b2,...]         # só partidas em que o R2D2 jogou com esses commits

Definições:
  - Sumô: cada evento "Empate (tempo)" é uma rodada empatada; "hero_N venceu" é uma rodada ganha pelo
    lado de N (hero_0 = azul = lado a). A partida é vencida quando o placar final é nosso.
  - Rouba-bandeira: pegada = um robô passa de não carregar para carregar a bandeira (amostras da
    gravação). Toques nossos = adversários que nossos robôs seguraram, pelos eventos "hero_N segurou"
    do juiz (ou pela gravação, nas partidas que o site não fechou).
  - Partidas com status "rodando" (o site não fechou; todas contra a Corsa) entram com o placar da
    última amostra gravada.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
from collections import defaultdict

US = "g5"  # Grupo 4 · R2D2
ORDEM = ["TuoLing", "RTech Tabajara", "Nyx", "Corsa", "SoLaDaBot", "Tech", "Claude"]
N_TIME = {"sumo": 1, "ctf": 2, "soccer": 3}  # robôs por time; os do lado a vêm primeiro


def ler(path):
    out = []
    with gzip.open(path, "rt") as f:
        for line in f:
            try:
                o = json.loads(line)
            except ValueError:
                continue
            if "t" in o and o.get("robots") and "match" in o:
                out.append(o)
    out.sort(key=lambda o: o["t"])
    dedup = []
    for o in out:  # gravações divididas por reinício repetem amostras
        if not dedup or o["t"] > dedup[-1]["t"] + 1e-6:
            dedup.append(o)
    return dedup


def partidas(src, ate=None, commits=None, excluir=()):
    equipes = {e["id"]: e["curto"] for e in json.load(open(os.path.join(src, "equipes.json")))}
    out = []
    for m in json.load(open(os.path.join(src, "partidas.json"))).values():
        if US not in (m["a"], m["b"]) or m["status"] not in ("concluida", "rodando"):
            continue
        g = m["_jogo"]
        arq = os.path.join(src, g, f"{m['id']}.jsonl.gz")
        if not os.path.exists(arq):
            continue
        res = m.get("resultado") or resultado_da_gravacao(arq)  # "rodando": o site não fechou a partida
        if ate and m["inicio"][:len(ate)] > ate:
            continue
        lado = 0 if m["a"] == US else 1
        commit = (m.get("versoes") or {}).get("ab"[lado], {}).get("commit")
        if commits and commit not in commits:
            continue
        if f"{g}:{commit}" in excluir:
            continue
        out.append(dict(jogo=g, id=m["id"], lado=lado, arq=arq, res=res, commit=commit,
                        concluida=m["status"] == "concluida",
                        adv=equipes[m["b"] if lado == 0 else m["a"]]))
    return out


def resultado_da_gravacao(arq):
    """Placar da última amostra e as mensagens do juiz, para partidas que o site não fechou."""
    S = ler(arq)
    ev, prev = [], ""
    for o in S:
        msg = o.get("message") or ""
        if msg and msg != prev:
            ev.append({"t": o["match"].get("clock", o["t"]), "msg": msg, "score": o["match"]["score"]})
        prev = msg
    return {"score": S[-1]["match"]["score"] if S else [0, 0], "events": ev}


def lado_do_robo(i, jogo):
    return 0 if i < N_TIME[jogo] else 1


# ---------------- por jogo ----------------

def sumo(p):
    v = e = d = 0
    t_vit = []
    t0 = 0.0
    for ev in p["res"].get("events") or []:
        msg = ev["msg"]
        if msg.startswith("Empate"):
            e += 1
        else:
            m = re.match(r"hero_(\d) venceu", msg)
            if not m:
                continue
            if lado_do_robo(int(m.group(1)), "sumo") == p["lado"]:
                v += 1
                t_vit.append(ev["t"] - t0)
            else:
                d += 1
        t0 = ev["t"] + 2.0  # o juiz reposiciona por ~2 s antes da rodada seguinte
    s = p["res"]["score"]
    venceu = s[p["lado"]] > s[1 - p["lado"]]
    return dict(rv=v, re=e, rd=d, venceu=int(venceu), t_vit=t_vit)


def ctf(p):
    S = ler(p["arq"])
    peg, toq = [0, 0], [0, 0]
    prev_c, prev_h = None, None
    for o in S:
        c = list(o.get("carrying") or [False] * 4)[:4]
        h = [False] * 4
        for r in o["robots"]:
            if 0 <= r["id"] < 4:
                h[r["id"]] = bool(r.get("held"))
        if prev_c is not None:
            for i in range(4):
                time = lado_do_robo(i, "ctf")
                if c[i] and not prev_c[i]:
                    peg[time] += 1
                if h[i] and not prev_h[i]:
                    toq[time] += 1
        prev_c, prev_h = c, h
    ev = [re.match(r"hero_(\d) segurou", e["msg"]) for e in p["res"].get("events") or []]
    ev = [int(m.group(1)) for m in ev if m]
    if ev and p["concluida"]:  # o juiz registra cada toque; as amostras perdem os que caem num buraco
        toq = [sum(lado_do_robo(i, "ctf") == t for i in ev) for t in (0, 1)]
    else:  # na gravação, toq[t] conta robôs do time t segurados, ou seja, toques do outro time
        toq = toq[::-1]
    s = p["res"]["score"]
    L = p["lado"]
    # toques nossos = adversários que seguramos
    return dict(pn=s[L], pd=s[1 - L], gn=peg[L], gd=peg[1 - L], tn=toq[L], td=toq[1 - L])


def soccer(p):
    s = p["res"]["score"]
    L = p["lado"]
    return dict(gn=s[L], gd=s[1 - L])


# ---------------- saída ----------------

def agrupa(ps, f):
    por = defaultdict(list)
    for p in ps:
        por[p["adv"]].append(f(p))
    return [(a, por[a]) for a in ORDEM if a in por]


def num(x, casas=2):
    return f"{x:.{casas}f}".replace(".", "{,}")


def escreve(path, linhas):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")
    print("escrito", path)


def tab_sumo(G, path):
    L, tot = [], [0] * 5
    for a, rs in G:
        row = [len(rs), sum(r["venceu"] for r in rs), sum(r["rv"] for r in rs),
               sum(r["re"] for r in rs), sum(r["rd"] for r in rs)]
        tot = [x + y for x, y in zip(tot, row)]
        L.append(f"{a} & " + " & ".join(map(str, row)) + r" \\")
    L += [r"\hline", "Total & " + " & ".join(map(str, tot)) + r" \\"]
    escreve(path, L)
    tv = [t for _, rs in G for r in rs for t in r["t_vit"]]
    print(f"  sumô: {tot[2] + tot[3] + tot[4]} rodadas, {tot[3]} empates; "
          f"duração média das {len(tv)} vitórias: {sum(tv) / max(len(tv), 1):.1f} s")


def tab_ctf(G, path):
    k = ["pn", "pd", "gn", "gd", "tn", "td"]
    L, tot, n = [], dict.fromkeys(k, 0), 0
    for a, rs in G:
        n += len(rs)
        row = {x: sum(r[x] for r in rs) for x in k}
        for x in k:
            tot[x] += row[x]
        L.append(f"{a} & {len(rs)} & " + " & ".join(str(row[x]) for x in k) + r" \\")
    L += [r"\hline", f"Total & {n} & " + " & ".join(str(tot[x]) for x in k) + r" \\"]
    escreve(path, L)


def tab_soccer(G, path):
    L, n, gn, gd = [], 0, 0, 0
    for a, rs in G:
        n += len(rs)
        gn += sum(r["gn"] for r in rs)
        gd += sum(r["gd"] for r in rs)
        L.append(f"{a} & {len(rs)} & {sum(r['gn'] for r in rs)} & {sum(r['gd'] for r in rs)}" + r" \\")
    L += [r"\hline", f"Total & {n} & {gn} & {gd}" + r" \\"]
    escreve(path, L)


def tab_rodada_final(csv_path, commit, path):
    """Rodada de 27/11 (versões finais RL), pelo partidas.csv do site: V/E/D e placar por adversário."""
    import csv
    por = defaultdict(lambda: {g: [0, 0, 0, 0, 0] for g in ("sumo", "ctf", "soccer")})
    for r in csv.DictReader(open(csv_path)):
        if r["status"] != "concluida" or US not in (r["a"], r["b"]) or commit not in (r["commit_a"], r["commit_b"]):
            continue
        us, th = ("a", "b") if r["a"] == US else ("b", "a")
        pn, pd = int(r["placar_" + us]), int(r["placar_" + th])
        adv = r["nome_" + th].split("· ")[-1].replace("Claude (agente da disciplina)", "Claude")
        t = por[adv][r["jogo"]]
        t[0 if pn > pd else (2 if pn < pd else 1)] += 1
        t[3] += pn
        t[4] += pd
    tot = {g: [0] * 5 for g in ("sumo", "ctf", "soccer")}
    L = []
    for a in ORDEM:
        if a not in por:
            continue
        cel = []
        for g in ("sumo", "ctf", "soccer"):
            v = por[a][g]
            tot[g] = [x + y for x, y in zip(tot[g], v)]
            cel.append(f"{v[0]}/{v[1]}/{v[2]}" + ("" if g == "sumo" else f", {v[3]} $\\times$ {v[4]}") if sum(v[:3]) else "não jogou")
        L.append(f"{a} & " + " & ".join(cel) + r" \\")
    cel = [f"{tot[g][0]}/{tot[g][1]}/{tot[g][2]}" + ("" if g == "sumo" else f", {tot[g][3]} $\\times$ {tot[g][4]}") for g in ("sumo", "ctf", "soccer")]
    L += [r"\hline", "Total & " + " & ".join(cel) + r" \\"]
    escreve(path, L)


def figuras(Gs, d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    VERDE, CINZA, VERM = "#2e7d4f", "#b0b0b0", "#c0392b"
    os.makedirs(d, exist_ok=True)

    def barras(ax, nomes, series, cores, rot):
        y = range(len(nomes))
        esq = [0] * len(nomes)
        for vals, c, r in zip(series, cores, rot):
            ax.barh(y, vals, left=esq, color=c, label=r, height=0.65)
            for i, (e0, v) in enumerate(zip(esq, vals)):
                if v:
                    ax.text(e0 + v / 2, i, str(v), ha="center", va="center", fontsize=7,
                            color="white" if c != CINZA else "black")
            esq = [a + b for a, b in zip(esq, vals)]
        ax.set_yticks(list(y), nomes)
        ax.invert_yaxis()

    G = Gs["sumo"]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    barras(ax, [a for a, _ in G], [[sum(r[k] for r in rs) for _, rs in G] for k in ("rv", "re", "rd")],
           [VERDE, CINZA, VERM], ["vitória", "empate", "derrota"])
    ax.set_xlabel("rodadas")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "torneio_sumo.pdf"))
    plt.close(fig)

    G = Gs["ctf"]
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
    nomes = [a for a, _ in G]
    for ax, (kn, kd, tit) in zip(axs, [("gn", "gd", "pegadas da bandeira"), ("tn", "td", "toques (adversários segurados)")]):
        y = range(len(nomes))
        ax.barh([i - 0.18 for i in y], [sum(r[kn] for r in rs) for _, rs in G], 0.36, color=VERDE, label="R2D2")
        ax.barh([i + 0.18 for i in y], [sum(r[kd] for r in rs) for _, rs in G], 0.36, color=VERM, label="adversário")
        ax.set_yticks(list(y), nomes)
        ax.set_title(tit, fontsize=9)
    axs[0].invert_yaxis()
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "torneio_ctf.pdf"))
    plt.close(fig)

    G = Gs["soccer"]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    nomes = [a for a, _ in G]
    y = range(len(nomes))
    ax.barh([i - 0.18 for i in y], [sum(r["gn"] for r in rs) for _, rs in G], 0.36, color=VERDE, label="nossos")
    ax.barh([i + 0.18 for i in y], [sum(r["gd"] for r in rs) for _, rs in G], 0.36, color=VERM, label="deles")
    ax.set_yticks(list(y), nomes)
    ax.invert_yaxis()
    ax.set_xlabel("gols")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "torneio_soccer.pdf"))
    plt.close(fig)
    print("figuras em", d)


def main():
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(raiz, "ultimas_tentativas", "gravacoes_ao_vivo"))
    ap.add_argument("--tabelas", default=os.path.join(raiz, "overleaf", "tabelas"))
    ap.add_argument("--figuras", default=os.path.join(raiz, "figuras_torneio"))
    ap.add_argument("--ate", help='só partidas iniciadas até "AAAA-MM-DD HH:MM"')
    ap.add_argument("--commits", help="só partidas com esses commits do R2D2 (separados por vírgula)")
    ap.add_argument("--excluir", default="", help='partidas a deixar de fora, como "ctf:deb2374,sumo:*"')
    ap.add_argument("--sem-figuras", action="store_true")
    a = ap.parse_args()
    exc = set(x for x in a.excluir.split(",") if x)
    ps = partidas(a.src, a.ate, set(a.commits.split(",")) if a.commits else None, exc)
    por = defaultdict(list)
    for p in ps:
        por[p["jogo"]].append(p)
    print({g: len(v) for g, v in por.items()})
    Gs = {"sumo": agrupa(por["sumo"], sumo), "ctf": agrupa(por["ctf"], ctf),
          "soccer": agrupa(por["soccer"], soccer)}
    tab_sumo(Gs["sumo"], os.path.join(a.tabelas, "torneio_sumo.tex"))
    pcsv = os.path.join(raiz, "torneio_logs", "dados", "partidas.csv")
    if os.path.exists(pcsv):
        tab_rodada_final(pcsv, "5acc156", os.path.join(a.tabelas, "torneio_final.tex"))
    tab_ctf(Gs["ctf"], os.path.join(a.tabelas, "torneio_ctf.tex"))
    tab_soccer(Gs["soccer"], os.path.join(a.tabelas, "torneio_soccer.tex"))
    if not a.sem_figuras:
        figuras(Gs, a.figuras)


if __name__ == "__main__":
    main()
