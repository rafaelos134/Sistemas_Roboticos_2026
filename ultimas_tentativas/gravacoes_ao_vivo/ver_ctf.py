#!/usr/bin/env python3
"""Visualizador rápido das partidas de rouba-bandeira gravadas ao vivo no torneio PSR.

Uso:
  python3 ver_ctf.py                 # lista as partidas (partidas.csv)
  python3 ver_ctf.py 641             # abre partidas/641.jsonl.gz
  python3 ver_ctf.py 641 --desde 60  # começa no segundo 60
  python3 ver_ctf.py caminho/arquivo.jsonl.gz
  python3 ver_ctf.py R2D2            # lista só as partidas de uma equipe

Controles: espaço = play/pausa · ←/→ = ±2 s · ,/. = quadro a quadro
           ↑/↓ = velocidade · r = rastro on/off · barra = ir para um instante
Requer: matplotlib (pip install matplotlib)

Cores: lado azul (robôs 0,1 no 2v2 / 0 no 1v1) defende a bandeira da esquerda (x<0),
lado vermelho (2,3 / 1) a da direita. Robô preso = X cinza; quem carrega a bandeira
ganha um anel na cor da bandeira inimiga.
"""
import argparse, csv, gzip, json, math, os, sys

AQUI = os.path.dirname(os.path.abspath(__file__))
AZUL, VERMELHO = "#2b6cd6", "#d63b3b"


def ler(path):
    cab, amostras = {}, []
    abrir = gzip.open if path.endswith(".gz") else open
    with abrir(path, "rt") as f:
        for linha in f:
            try:
                o = json.loads(linha)
            except ValueError:
                continue  # linha cortada (gravação interrompida)
            if o.get("tipo") == "inicio":
                cab = cab or o
            elif "t" in o and o.get("robots"):
                amostras.append(o)
    amostras.sort(key=lambda o: o["t"])
    return cab, amostras


def listar(filtro=None):
    path = os.path.join(AQUI, "ctf", "partidas.csv")
    if not os.path.exists(path):
        sys.exit("partidas.csv não encontrado; passe o caminho de um .jsonl.gz")
    with open(path) as f:
        linhas = list(csv.DictReader(f))
    if filtro:
        linhas = [l for l in linhas if filtro.lower() in (l["azul"] + l["vermelho"]).lower()]
    print(f"{'id':>5}  {'data':16}  {'fase':10} {'placar':>7}  azul × vermelho")
    for l in linhas:
        print(f"{l['partida']:>5}  {l['data']:16}  {l['fase']:10} {l['placar_azul']:>3}:{l['placar_vermelho']:<3}  "
              f"{l['azul']} × {l['vermelho']}")
    print(f"\n{len(linhas)} partidas. Abra uma com: python3 ver_ctf.py <id>")


def equipes(pid):
    try:
        with open(os.path.join(AQUI, "ctf", "partidas.csv")) as f:
            for l in csv.DictReader(f):
                if l["partida"] == str(pid):
                    return l["azul"], l["vermelho"]
    except OSError:
        pass
    return "azul", "vermelho"


def lado(rid, n):
    return 0 if (rid == 0 if n <= 2 else rid < 2) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("partida", nargs="?", help="id da partida, arquivo .jsonl.gz ou nome de equipe para filtrar")
    ap.add_argument("--desde", type=float, default=0.0, help="segundo inicial")
    ap.add_argument("--vel", type=float, default=1.0, help="velocidade inicial (1 = tempo real)")
    a = ap.parse_args()

    if a.partida is None or not (a.partida.isdigit() or os.path.exists(a.partida)):
        return listar(a.partida)
    path = a.partida if os.path.exists(a.partida) else os.path.join(AQUI, "ctf", f"{a.partida}.jsonl.gz")
    if not os.path.exists(path):
        sys.exit(f"não achei {path}")
    cab, S = ler(path)
    if not S:
        sys.exit("gravação vazia")
    pid = cab.get("partida") or os.path.basename(path).split(".")[0]
    nome_a, nome_b = equipes(pid)

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle, Polygon
    from matplotlib.widgets import Slider, Button

    T = [o["t"] for o in S]
    fig = plt.figure(figsize=(8.5, 9.2))
    fig.canvas.manager.set_window_title(f"CTF #{pid} — {nome_a} × {nome_b}")
    ax = fig.add_axes([0.05, 0.16, 0.9, 0.74])
    ax.set_aspect("equal")
    ax.set_xlim(-0.85, 0.85); ax.set_ylim(-0.85, 0.85)
    ax.set_xticks([]); ax.set_yticks([])
    # campo: metades tingidas, linha do meio, bases
    ax.add_patch(Rectangle((-0.8, -0.8), 0.8, 1.6, color=AZUL, alpha=0.07))
    ax.add_patch(Rectangle((0, -0.8), 0.8, 1.6, color=VERMELHO, alpha=0.07))
    ax.add_patch(Rectangle((-0.8, -0.8), 1.6, 1.6, fill=False, lw=2, color="#444"))
    ax.plot([0, 0], [-0.8, 0.8], color="#888", lw=1, ls="--")
    for x, c in ((-0.62, AZUL), (0.62, VERMELHO)):
        ax.add_patch(Circle((x, 0), 0.14, fill=False, lw=1.2, ls=":", color=c))

    titulo = fig.text(0.5, 0.955, "", ha="center", va="center", fontsize=15, weight="bold")
    fig.text(0.05, 0.955, nome_a, color=AZUL, ha="left", va="center", fontsize=10, weight="bold")
    fig.text(0.95, 0.955, nome_b, color=VERMELHO, ha="right", va="center", fontsize=10, weight="bold")
    info = fig.text(0.5, 0.92, "", ha="center", va="center", fontsize=9, color="#555")

    n = len(S[0]["robots"])
    corpos, setas, rotulos, aneis, presos, rastros = [], [], [], [], [], []
    for i in range(n):
        c = AZUL if lado(i, n) == 0 else VERMELHO
        corpos.append(ax.add_patch(Circle((0, 0), 0.037, color=c, zorder=5)))
        setas.append(ax.plot([], [], color="white", lw=2, zorder=6)[0])
        aneis.append(ax.add_patch(Circle((0, 0), 0.055, fill=False, lw=2.5, zorder=4, visible=False)))
        presos.append(ax.plot([], [], "x", color="#333", ms=14, mew=2.5, zorder=7)[0])
        rotulos.append(ax.text(0, 0, str(i), fontsize=8, ha="center", va="bottom", zorder=8))
        rastros.append(ax.plot([], [], color=c, lw=1, alpha=0.4, zorder=3)[0])
    bandeiras = {}

    def bandeira(team):
        if team not in bandeiras:
            c = AZUL if team == "blue" else VERMELHO
            mastro = ax.plot([], [], color="#222", lw=1.5, zorder=9)[0]
            pano = ax.add_patch(Polygon([[0, 0]] * 3, closed=True, color=c, zorder=9))
            bandeiras[team] = (mastro, pano)
        return bandeiras[team]

    est = {"i": 0, "play": False, "vel": a.vel, "rastro": True, "mudando": False}

    def idx(t):
        lo, hi = 0, len(T) - 1
        while lo < hi:
            m = (lo + hi + 1) // 2
            if T[m] <= t:
                lo = m
            else:
                hi = m - 1
        return lo

    def desenhar(i, mantem_t=False):
        est["i"] = i
        o = S[i]
        if not mantem_t:
            est["t"] = o["t"]
        carregando = o.get("carrying") or [False] * n
        for r in o["robots"]:
            k = r["id"]
            if k >= n:
                continue
            x, y, th = r["x"], r["y"], r.get("yaw", 0.0)
            corpos[k].center = (x, y)
            corpos[k].set_alpha(0.4 if r.get("held") else 1.0)
            setas[k].set_data([x, x + 0.045 * math.cos(th)], [y, y + 0.045 * math.sin(th)])
            rotulos[k].set_position((x, y + 0.045))
            presos[k].set_data([x] if r.get("held") else [], [y] if r.get("held") else [])
            leva = k < len(carregando) and carregando[k]
            aneis[k].set_visible(bool(leva))
            if leva:
                aneis[k].center = (x, y)
                aneis[k].set_edgecolor(VERMELHO if lado(k, n) == 0 else AZUL)
            if est["rastro"]:
                j0 = idx(o["t"] - 3.0)
                pts = [(rr["x"], rr["y"]) for s in S[j0:i + 1] for rr in s["robots"] if rr["id"] == k]
                rastros[k].set_data([p[0] for p in pts], [p[1] for p in pts])
            else:
                rastros[k].set_data([], [])
        vistas = set()
        for mk in o.get("markers") or []:
            if mk.get("kind") != "flag":
                continue
            vistas.add(mk.get("team"))
            mastro, pano = bandeira(mk.get("team"))
            x, y = mk["x"], mk["y"]
            mastro.set_data([x, x], [y, y + 0.09])
            pano.set_xy([[x, y + 0.09], [x + 0.06, y + 0.07], [x, y + 0.05]])
            mastro.set_visible(True); pano.set_visible(True)
        for team, (mastro, pano) in bandeiras.items():
            if team not in vistas:
                mastro.set_visible(False); pano.set_visible(False)
        m = o.get("match") or {}
        sc = m.get("score") or ["–", "–"]
        titulo.set_text(f"{sc[0]}  :  {sc[1]}")
        clock, dur = m.get("clock") or 0, m.get("duration") or 0
        info.set_text(f"#{pid} · {m.get('phase', '')} · {clock:5.1f}/{dur:.0f} s · {est['vel']:g}×"
                      + (f"  —  {o['message']}" if o.get("message") else ""))
        if not est["mudando"]:
            est["mudando"] = True
            barra.set_val(o["t"])
            est["mudando"] = False
        fig.canvas.draw_idle()

    axb = fig.add_axes([0.17, 0.08, 0.76, 0.03])
    barra = Slider(axb, "t (s)", T[0], T[-1], valinit=T[0])

    def ao_mover(v):
        if not est["mudando"]:
            est["mudando"] = True
            desenhar(idx(v))
            est["mudando"] = False
    barra.on_changed(ao_mover)

    def alternar(_=None):
        est["play"] = not est["play"]
        bplay.label.set_text("❚❚ pausa" if est["play"] else "▶ play")
        fig.canvas.draw_idle()

    def mudar_vel(f):
        est["vel"] = min(16, max(0.25, est["vel"] * f))
        desenhar(est["i"])

    bplay = Button(fig.add_axes([0.05, 0.02, 0.14, 0.045]), "▶ play")
    bplay.on_clicked(alternar)
    blento = Button(fig.add_axes([0.21, 0.02, 0.09, 0.045]), "« 0.5×")
    blento.on_clicked(lambda _: mudar_vel(0.5))
    brapido = Button(fig.add_axes([0.31, 0.02, 0.09, 0.045]), "2× »")
    brapido.on_clicked(lambda _: mudar_vel(2))
    fig._botoes = (bplay, blento, brapido)  # sem referência o matplotlib descarta os botões
    fig.text(0.43, 0.043, "espaço play · ←→ ±2 s · ,. quadro · ↑↓ velocidade · r rastro",
             fontsize=8, color="#666", va="center")

    def tecla(ev):
        i = est["i"]
        if ev.key == " ":
            alternar()
        elif ev.key == "right":
            desenhar(idx(S[i]["t"] + 2))
        elif ev.key == "left":
            desenhar(idx(S[i]["t"] - 2))
        elif ev.key == ".":
            desenhar(min(i + 1, len(S) - 1))
        elif ev.key == ",":
            desenhar(max(i - 1, 0))
        elif ev.key == "up":
            mudar_vel(2)
        elif ev.key == "down":
            mudar_vel(0.5)
        elif ev.key == "r":
            est["rastro"] = not est["rastro"]
            desenhar(i)
    fig.canvas.mpl_connect("key_press_event", tecla)

    PASSO = 0.05  # s de relógio real entre quadros

    def tick():
        if not est["play"]:
            return
        if est["i"] >= len(S) - 1:
            return alternar()
        # relógio próprio da reprodução: avança mesmo quando o passo é menor que o intervalo entre amostras
        est["t"] = max(est.get("t", 0), S[est["i"]]["t"]) + PASSO * est["vel"]
        j = idx(est["t"])
        if j != est["i"]:
            desenhar(j, mantem_t=True)
    timer = fig.canvas.new_timer(interval=int(PASSO * 1000))
    timer.add_callback(tick)
    timer.start()

    desenhar(idx(T[0] + a.desde) if a.desde else 0)
    plt.show()


if __name__ == "__main__":
    main()
