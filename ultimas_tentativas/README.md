# Últimas tentativas — dados do torneio, modelos de RL e o Nyx estimado

Material para quem quiser continuar o trabalho do R2D2 (Grupo 4): analisar as partidas reais do torneio,
treinar de novo a partir dos nossos modelos ou do zero, e testar contra adversários reais.

```
ultimas_tentativas/
├── gravacoes_ao_vivo/   todas as partidas gravadas ao vivo no site do torneio (sumô, rouba-bandeira, futebol)
├── rl/                  código de treino + melhores modelos dos 3 jogos + nós exportados + documentação
└── nyx_estimado/        engenharia reversa do rouba-bandeira do Nyx (regras, nó, validação)
```

## gravacoes_ao_vivo/

Fluxo `api/ao-vivo` do site do torneio, ~9 amostras/s, **todas as equipes**: posição e orientação de todos
os robôs, bola / bandeiras, quem está preso, quem carrega, placar, relógio e mensagens do juiz.

- `sumo/`, `ctf/`, `soccer/` → `<id>.jsonl.gz`, uma partida por arquivo (1ª linha = cabeçalho com o layout
  do campo; demais = amostras). Formato detalhado: docstring de `rl/live_ctf.py` (e `rl/replay_ctf_sim.py`).
- `partidas.csv` → índice: jogo, id, data, fase, azul × vermelho, placar, nº de amostras.
- `partidas.json`, `equipes.json` → metadados do torneio (times, resultados, eventos).

Ver as partidas:
```bash
cd gravacoes_ao_vivo
python3 replay_server.py          # abre http://localhost:8765 : os 3 jogos, play/pausa, velocidade, eventos
python3 ver_ctf.py R2D2           # só rouba-bandeira, matplotlib (lista as nossas); ver_ctf.py 672 abre a 672
```
Lado azul = lado `a` da partida (robôs 0,1 no 2×2; no futebol 0–2), vermelho = lado `b`.

## rl/

Cópia da pasta `rl/` do nosso ambiente (`trabalho_final_rl/ros_humble_base/local_mount/rl`): treino PPO em
paralelo no Genesis (GPU), exportação para um nó ROS de arquivo único (numpy) e avaliação no psr_sim.
**Precisa do psr-hero-kit** ao lado (`../kit/sim/psr_sim`): coloque esta pasta como `rl/` dentro do
`local_mount` do container (onde está o `kit/`), ou ajuste o caminho no topo de `ctf_env.py`,
`sumo_env.py`, `soccer_env.py` e `eval_sumo.py`. Documentação completa: `rl/docs/documentacao.md`.

Melhores modelos (`rl/runs/`, lista em `rl/runs/MODELOS.md`) e os nós prontos (`rl/exports/`):

| Jogo | Modelo | Nó exportado (o que roda no kit) |
|---|---|---|
| Sumô | `sumo_ppo_v5/model.pt` (iteração 1200) | `exports/sumo/sumo_rl_v5.py` = `projeto_final/rl/sumo.py` |
| Futebol | `soccer_att_a2w/model.pt` (+ goleiro de script) | `exports/soccer/soccer_rl.py` = `projeto_final/rl/soccer.py` |
| Rouba-bandeira | `ctf_att_4/model.pt` + defensor de script | `exports/ctf/ctf_hibrido_v4.py` = `projeto_final/rl/ctf.py` |
| Rouba-bandeira (novo) | `ctf_att_real_1/model.pt`: atacante treinado contra as **imitações dos times reais** | `exports/ctf/ctf_hibrido_real1.py` |

Também: os pools de treino do CTF (`ctf_att_1..5`, `ctf_def_1..5`, `ctf_team_2`), as imitações de cada
time real (`bc_<time>_att|def`, behavior cloning das gravações) e as avaliações
(`eval_live_ctf_att_4.jsonl`, `eval_live_ctf_att_real_1.jsonl`).

Continuar do que foi feito ou treinar do zero (dentro do container, `cd ~/ros2_ws/rl`):
```bash
PY=~/ros2_ws/.venv/bin/python
# rouba-bandeira com os adversários reais (dados -> imitações -> PPO -> avaliação), partindo do melhor:
mkdir -p data/ctf_ao_vivo && cp -r <ultimas_tentativas>/gravacoes_ao_vivo/ctf data/ctf_ao_vivo/partidas && cp data/ctf_ao_vivo/partidas/partidas.csv data/ctf_ao_vivo/
./train_live_ctf.sh ctf_att_real_2 400 runs/ctf_att_real_1/model.pt
# do zero: liga alternada atacante x defensor (ctf_league.sh), sumô (run_v5.sh), futebol (train_soccer.py):
./ctf_league.sh 3 500 400
# exportar para o kit:
$PY export_ctf.py --att runs/ctf_att_real_2/model.pt --def-script --out exports/ctf/ctf_hibrido_real2.py
$PY export_sumo.py runs/sumo_ppo_v5/model.pt exports/sumo/sumo_rl.py
$PY export_soccer.py --att runs/soccer_att_a2w/model.pt --out exports/soccer/soccer_rl.py
# ver um modelo contra uma partida real gravada (no VNC):
$PY vs_replay_ctf.py 672 --nosso exports/ctf/ctf_hibrido_real1.py
$PY replay_ctf_sim.py 672                     # só reproduzir a partida no simulador 3D
```

### Rouba-bandeira contra as imitações dos times reais (pontos por rodada de 90 s, simulador)

| Defensor imitado | `ctf_att_4` (antes) | `ctf_att_real_1` (novo) |
|---|---|---|
| Claude (agente da disciplina) | 0,55 | 4,46 |
| Grupo 1 · TuoLing | 0,29 | 1,82 |
| Grupo 2 · RTech Tabajara | 1,36 | 4,25 |
| Grupo 3 · Nyx | 0,20 | 2,87 |
| Grupo 5 · Corsa | 2,94 | 4,62 |
| Grupo 6 · SoLaDaBot | 1,23 | 5,87 |
| Grupo 7 · Tech | 0,25 | 0,51 |

Avaliado contra as mesmas imitações usadas no treino: é otimista. O teste de verdade é o torneio.

## nyx_estimado/

Engenharia reversa do rouba-bandeira do Nyx a partir dos trajetos: regras encontradas, nó no formato do
kit, scripts da análise e validação. Ver `nyx_estimado/README.md`.

## Atualizar

As gravações continuam sendo feitas em `trabalho_final_rl/torneio_logs` (gravar_ao_vivo.py). Para trazer as
novas e os modelos mais recentes para esta pasta: `python3 trabalho_final_rl/torneio_logs/exportar_ultimas.py`.
