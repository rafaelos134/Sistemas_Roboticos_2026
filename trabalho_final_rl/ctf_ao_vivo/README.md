# Rouba-bandeira — partidas gravadas ao vivo do torneio

Gravações do fluxo `api/ao-vivo` do site do torneio PSR (~9 amostras/s), com **todas as equipes**:
posição e orientação de todos os robôs, bandeiras, quem está preso, quem carrega a bandeira, placar,
relógio e mensagens do juiz.

- `partidas.csv` — índice: id, data, fase, equipe azul × vermelha, placar, nº de amostras.
- `partidas/<id>.jsonl.gz` — uma partida por arquivo.
- `ver_ctf.py` — visualizador rápido (matplotlib).

## Ver uma partida

```bash
pip install matplotlib
python3 ver_ctf.py            # lista as partidas
python3 ver_ctf.py R2D2       # lista só as nossas
python3 ver_ctf.py 672        # abre a partida 672
python3 ver_ctf.py 672 --desde 60 --vel 2
```

Controles: **espaço** play/pausa · **← →** ±2 s · **, .** quadro a quadro · **↑ ↓** velocidade ·
**r** rastro · barra de tempo para pular.

Lado **azul** (robôs 0,1) defende a bandeira da esquerda (x < 0); **vermelho** (2,3) a da direita.
Robô preso aparece com um X; quem carrega a bandeira ganha um anel na cor da bandeira inimiga.

## Formato (para usar em treino)

Cada linha é um JSON. A primeira é o cabeçalho (`{"tipo":"inicio", "partida", "layout", ...}`;
se a gravação foi interrompida pode haver mais de um — basta ignorar linhas com `"tipo"`).
As demais são amostras:

```json
{"ts": 1795609774.6, "t": 21.5,
 "robots": [{"id":0,"x":-0.47,"y":0.22,"yaw":0.0,"held":false}, ...],
 "markers": [{"kind":"flag","x":-0.62,"y":0.0,"team":"blue"}, {"kind":"flag","team":"red", ...}],
 "carrying": [false,false,true,false],
 "match": {"phase":"jogo","clock":16.8,"duration":180.0,"score":[0,0]},
 "message": "hero_3 segurou o hero_0 por 4 s"}
```

- `t`: tempo do simulador (inclui o aquecimento); `match.clock`: relógio da partida.
- `phase`: `aquecimento` → `jogo` → fim.
- Campo 1,6 × 1,6 m centrado em (0,0); bases em x = ±0,62 (raio 0,14).
- Não há comandos de velocidade nem quem é atacante/defensor — só o estado observado.

Leitura em Python:

```python
import gzip, json
amostras = [o for o in map(json.loads, gzip.open("partidas/672.jsonl.gz", "rt")) if "t" in o]
```
