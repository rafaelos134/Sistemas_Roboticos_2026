# Projeto final: sumô, rouba-bandeira e futebol (Grupo 4, R2D2)

Cada jogo tem duas estratégias. A **linha RL** (PPO, com os pesos embutidos no arquivo e calculados em
`numpy`) é a versão final entregue, e a **linha clássica** (máquinas de estados com controladores
proporcionais) é a referência de comparação. Todos os nós usam só `numpy`, `rospy`, as mensagens ROS e
`psr_hero.utils`, e o mesmo arquivo joga dos dois lados do campo.

## Arquivos

| Jogo | Final (RL) | Clássica | Versões anteriores usadas nas comparações |
|---|---|---|---|
| Sumô | `sumo/sumo.py`, rede da fase 6 com EKF | `sumo/sumo_classico.py` (commit `55563fb`) | `sumo/anteriores/sumo_rl_fase5.py` (RL anterior), `sumo/anteriores/sumo_classico_simples.py` (clássica versão 1, commit `401b601`) |
| Rouba-bandeira | `ctf/ctf.py`, atacante PPO e defensor por regras | `ctf/ctf_classico.py` (commits `cb14cf5` a `72ca51b`) | `ctf/anteriores/ctf_hibrido_v4.py` (RL anterior), `ctf/anteriores/ctf_classico_anterior.py` (clássica versão 1, commit `deb2374`), `ctf/anteriores/ctf_rl_completo.py` (variante com atacante e defensor de RL) |
| Futebol | `soccer/soccer.py`, goleiro por regras e dois atacantes PPO (A2w) | `soccer/soccer_classico.py` (commit `403f030`) | `soccer/anteriores/soccer_rl_a2.py` (RL anterior, A2), `soccer/anteriores/soccer_script.py` (time por regras usado no treino) |

Os arquivos finais são cópias de `projeto_final/rl/` e `projeto_final/classico/`.
`sumo/anteriores/sumo_rl_v3.py` é a rede que jogou o sumô na rodada de 25/11 do torneio.

## Como rodar no kit

Copie os arquivos do jogo para `psr-hero-kit/exercicios/<jogo>/` e, na raiz do kit:

```bash
./run.sh run sumo   robot0:=sumo.py robot1:=sumo_classico.py
./run.sh run soccer blue:=soccer.py  red:=soccer_classico.py
./run.sh run ctf    blue:=ctf.py     red:=ctf_classico.py
```

Para jogar contra o time parado, use `robot1:=none` ou `red:=none`. Para trocar de lado, inverta os
arquivos.

## Registros (`registros/`)

- `simulador/classico/`: partidas da linha clássica no `psr_sim`. O `avaliacoes.log` tem o sumô (10
  rodadas de cada lado contra o robô parado, a versão 1 e a RL) e o rouba-bandeira; os CSV têm uma
  linha por partida de 5 min, com o placar e o instante de cada gol.
- `simulador/rl_final/`: partidas das versões finais RL.
  - `sumo_final.txt`: 20 rodadas de cada lado contra o robô parado, o `sumo_chaser.py`, o
    `sumo_flanker.py`, a fase 3, a fase 5 e a clássica, sem ruído (`SEMRUIDO`) e com o ruído do kit
    (`RUIDOKIT`).
  - `ctf_sem_ruido/`, `ctf_com_ruido/` e `ctf_entrega/`: séries de 5 min do rouba-bandeira contra o
    time parado, a clássica, o `ctf_script.py`, o RL completo e o híbrido RL v4.
  - `soccer_*.csv`: futebol contra o time parado, o A2 e a clássica, dos dois lados.
- `torneio/partidas_r2d2.csv`: as partidas da equipe no torneio, com o placar, o lado e o commit em
  jogo. `torneio/logs/<jogo>/<id>.txt` é o log do nosso nó em cada partida.
- As tabelas e figuras do torneio no relatório saem de `../scripts/analise_torneio.py`, que lê as
  gravações ao vivo do site do torneio.

## Vídeos (`videos/`)

Um vídeo curto por jogo e por estratégia, gravado no `psr_sim` com o placar na tela. A lista está em
`videos/LEIAME.txt`.

## Refazer as avaliações

As avaliações usam os avaliadores `eval_sumo.py`, `eval_soccer.py` e `eval_ctf.py` da pasta `rl/` do
ambiente de treino, que importam o nó com o `rospy` substituído e o chamam a 20 Hz no `psr_sim`.
Os scripts de `avaliacao/` supõem esta pasta em `~/ros2_ws/exercicios` e os avaliadores em
`~/ros2_ws/rl`. Dentro do container:

```bash
cd ~/ros2_ws/exercicios/avaliacao && ./rodar_avaliacoes.sh && ./rodar_avaliacoes_2.sh
```

Exemplos avulsos:

```bash
$PY eval_sumo.py ../exercicios/sumo/sumo.py --opponent ../exercicios/sumo/sumo_classico.py --rounds 20 --side 0
$PY eval_ctf.py  --blue ../exercicios/ctf/ctf.py --red ../exercicios/ctf/anteriores/ctf_hibrido_v4.py --minutes 5 --matches 3
$PY eval_soccer.py --blue ../exercicios/soccer/soccer.py --minutes 5 --matches 3
```

- `avaliacao/_game_wrapper.py` e `avaliacao/eval_ctf_classico*.py`: o avaliador de rouba-bandeira não
  chama `on_game()`, e o invólucro repassa o `/game/state` ao nó clássico sem mudar a estratégia.
- `avaliacao/eval_soccer_classico.py`: copia `ball_v` para `ball_vel`, o campo que o nó clássico lê.
- `avaliacao/soccer_parado.py`: time parado, para jogar dos dois lados.
