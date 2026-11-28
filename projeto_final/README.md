# Projeto final: sumô, futebol e rouba-bandeira (entrega)

Grupo 4 · R2D2. Esta pasta é a **entrega organizada** do projeto final: as duas famílias de soluções (clássica e
aprendizado por reforço) e o código usado para treinar as versões de RL. Ela **não** é lida pelo servidor do
torneio; a pasta que o servidor lê é a `../projeto-final/` (veja o README da raiz).

```
projeto_final/
├── classico/              # máquinas de estados escritas à mão
│   ├── sumo.py            # sumô clássico (aproximar por trás, contornar a ±60°, empurrar na radial, sacudir)
│   ├── soccer.py          # futebol clássico (um nó controla os 3 robôs)
│   └── ctf.py             # rouba-bandeira clássico (defensor sobre a bandeira, atacante que aceita ser segurado)
├── rl/                    # versões finais de RL (políticas PPO com os pesos embutidos no próprio arquivo)
│   ├── sumo.py            # sumo_ppo_v6_fus, iteração 800: MLP 12→64→64→2 + EKF odometria × lidar (kit com ruído)
│   ├── soccer.py          # atacantes soccer_att_a2w (MLP 28→128→128→2) + goleiro por script
│   └── ctf.py             # atacante ctf_zs_robusto (MLP 38→128→128→2) + defensor por script (ctf_script.py do kit)
└── treinamento/           # código de treino, exportação e avaliação das versões de rl/
    ├── treinar_sumo.sh            # sumô v6 (sensores com ruído + EKF), a partir da v5 incluída
    ├── treinar_futebol.sh         # A1 → A2 → A2w
    ├── treinar_rouba_bandeira.sh  # base → fuga → robusto
    ├── modelos/                   # config.json, checkpoint e log.csv das versões finais (sumô: v6 + ponto de partida v5)
    ├── eng_reversa/               # máquinas de estados estimadas dos times do torneio (adversários do rouba-bandeira)
    ├── documentacao.md            # documentação detalhada do treino (fases, recompensas, resultados)
    ├── eval_ctf_classico.py       # invólucro para avaliar o classico/ctf.py no eval_ctf.py
    └── *.py                       # ambientes, PPO, exportadores e avaliadores
```

Os nós de `classico/` e `rl/` são arquivos únicos: usam só Python padrão, `numpy`, `rospy`, as mensagens ROS e o
`psr_hero.utils`, e jogam dos dois lados (`~robot`/`~opponent` no sumô, `~team` no futebol, `~team`/`~role` no
rouba-bandeira). Para rodar um deles no kit, copie-o para a pasta do jogo (`exercicios/<jogo>/`) e use o `run.sh`
do kit, por exemplo `./run.sh run ctf blue:=ctf.py red:=none`.

## Versões de RL

As redes são treinadas na GPU, em milhares de cenários em paralelo (simulador Genesis com a física do `psr_sim`),
com PPO ator-crítico (`treinamento/ppo.py`). O nó final leva só o ator, exportado em base64, e o executa com
`numpy` a 20 Hz, de forma determinística (média da gaussiana).

| Jogo | Arquivo | Treino | Detalhes |
|---|---|---|---|
| Sumô | `rl/sumo.py` | `sumo_ppo_v6_fus` (v6) | parte do modelo da v5 e do grupo de cópias congeladas dela (`modelos/sumo_ppo_v6_fus/init_v5/`), com sensores ruidosos e EKF dos dois robôs (odometria × lidar); adversários: cópias, perseguidores insistentes e rodadas encostadas. Exportada a iteração 800 |
| Futebol | `rl/soccer.py` | `soccer_att_a1` → `soccer_att_a2` → `soccer_att_a2w` | atacantes de RL (uma rede para os dois), goleiro por script; A2w acrescenta o potencial "bola longe das paredes" |
| Rouba-bandeira | `rl/ctf.py` | `ctf_zs_base` → `ctf_zs_fuga` → `ctf_zs_robusto` | só o atacante é treinado; o defensor é sempre o script do kit. Adversários: parado, scripts com estilos e ganhos sorteados e as máquinas estimadas dos 8 times do torneio; a última fase acrescenta latência de comando |

Os comandos exatos de cada fase estão nos scripts `treinamento/treinar_*.sh` e os argumentos usados, em
`treinamento/modelos/<run>/config.json`.

## Como reproduzir o treino

Requisitos: GPU NVIDIA, Python 3.10, PyTorch, [Genesis](https://github.com/Genesis-Embodied-AI/Genesis), `numpy`, e o
`psr-hero-kit` (o treino e a avaliação usam o `psr_sim` do kit). O código procura o kit em `projeto_final/kit/`
(ou na variável `PSR_KIT`):

```bash
cd projeto_final
ln -s /caminho/para/psr-hero-kit kit        # ou: export PSR_KIT=/caminho/para/psr-hero-kit
cd treinamento
PY=python3 ./treinar_sumo.sh                # gera runs/… e exports/sumo.py
PY=python3 ./treinar_futebol.sh             # exports/soccer.py
PY=python3 ./treinar_rouba_bandeira.sh      # exports/ctf.py
```

Para só regenerar os nós a partir dos checkpoints incluídos:

```bash
python3 export_sumo.py modelos/sumo_ppo_v6_fus/it0800.pt exports/sumo.py --fusion
python3 export_soccer.py --att modelos/soccer_att_a2w/model.pt --no-unjam --out exports/soccer.py
python3 export_ctf.py --att modelos/ctf_zs_robusto/att.pt --def-script --out exports/ctf.py
```

O sumô e o rouba-bandeira saem idênticos, byte a byte, aos de `rl/`. O futebol sai com os mesmos pesos; o exportador
atual só acrescenta o código, desligado (`USE_FUSION = False`), da versão com fusão de sensores.

Avaliação no `psr_sim` (nós contra nós, mais rápido que o tempo real; `REAL_NOISE=1` aplica o ruído do kit da
competição):

```bash
python3 eval_sumo.py ../rl/sumo.py --opponent ../classico/sumo.py --rounds 10
python3 eval_ctf.py --blue ../rl/ctf.py --red eval_ctf_classico.py --minutes 5 --matches 3   # clássico pelo invólucro*
python3 eval_soccer.py --blue ../rl/soccer.py --red ../classico/soccer.py --minutes 5
```

\* O avaliador não chama o `on_game()`, do qual o rouba-bandeira clássico depende; o `eval_ctf_classico.py`
(com `_game_wrapper.py`) só repassa a ele o `/game/state` simulado, como faz a ponte ROS do kit.
