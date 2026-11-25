#!/usr/bin/env python3
"""Servidor local do visualizador de replays (replay.html).

Uso:
  python3 replay_server.py            # http://localhost:8765
  python3 replay_server.py --porta 9000

Pelo SSH: o VS Code encaminha a porta sozinho; ou, do seu computador,
  ssh -L 8765:localhost:8765 <servidor>   e abra http://localhost:8765

Rotas:
  /                         replay.html
  /api/partidas             lista das gravações (dados/ao_vivo) + times/placar de dados/partidas.json
  /api/replay?jogo=&id=     a gravação inteira em JSON lines (já descomprimida)
Só lê arquivos; escuta apenas em 127.0.0.1.
"""
import argparse, gzip, json, os, re
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

AQUI = os.path.dirname(os.path.abspath(__file__))
DADOS = AQUI
AO_VIVO = AQUI
JOGOS = ("sumo", "ctf", "soccer")


def carregar(path, padrao):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return padrao


def listar():
    partidas = carregar(os.path.join(DADOS, "partidas.json"), {})
    equipes = {e["id"]: e for e in carregar(os.path.join(DADOS, "equipes.json"), [])}
    out = []
    for jogo in JOGOS:
        d = os.path.join(AO_VIVO, jogo)
        if not os.path.isdir(d):
            continue
        for nome in os.listdir(d):
            m = re.fullmatch(r"(\d+)\.jsonl(\.gz)?", nome)
            if not m:
                continue
            pid, gz = int(m.group(1)), bool(m.group(2))
            if not gz and os.path.exists(os.path.join(d, nome + ".gz")):
                continue  # pedaço ainda aberto de uma partida que já tem .gz: lista uma vez só
            meta = partidas.get(f"{jogo}:{pid}", {})
            a, b = meta.get("a_eq") or meta.get("a"), meta.get("b_eq") or meta.get("b")
            r = meta.get("resultado") or {}
            st = os.stat(os.path.join(d, nome))
            out.append({
                "jogo": jogo, "id": pid, "ao_vivo": not gz, "kb": round(st.st_size / 1024),
                "gravado": st.st_mtime,
                "a": a, "b": b,
                "nome_a": (equipes.get(a) or {}).get("nome", a), "nome_b": (equipes.get(b) or {}).get("nome", b),
                "placar": r.get("score"), "status": meta.get("status"), "fase": meta.get("fase"),
                "inicio": meta.get("inicio"),
            })
    out.sort(key=lambda x: -x["gravado"])
    return out


def ler_replay(jogo, pid):
    if jogo not in JOGOS:
        return None
    base = os.path.join(AO_VIVO, jogo, f"{int(pid)}.jsonl")
    partes = []
    if os.path.exists(base + ".gz"):
        with gzip.open(base + ".gz", "rb") as f:  # lê também gzip multi-membro
            partes.append(f.read())
    if os.path.exists(base):
        with open(base, "rb") as f:
            partes.append(f.read())
    return b"".join(partes) if partes else None


class H(BaseHTTPRequestHandler):
    def enviar(self, corpo, tipo, status=200):
        self.send_response(status)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(corpo)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(corpo)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/replay.html"):
            with open(os.path.join(AQUI, "replay.html"), "rb") as f:
                return self.enviar(f.read(), "text/html; charset=utf-8")
        if u.path == "/api/partidas":
            return self.enviar(json.dumps(listar()).encode(), "application/json")
        if u.path == "/api/replay":
            q = parse_qs(u.query)
            try:
                corpo = ler_replay(q["jogo"][0], q["id"][0])
            except (KeyError, ValueError):
                corpo = None
            if corpo is None:
                return self.enviar(b"nao encontrado", "text/plain", 404)
            return self.enviar(corpo, "application/x-ndjson")
        self.enviar(b"nao encontrado", "text/plain", 404)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--porta", type=int, default=8765)
    a = ap.parse_args()
    print(f"replay em http://localhost:{a.porta}  (Ctrl+C para sair)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.porta), H).serve_forever()
