# TP5 — ROS e navegação (avoid e goto)

Projeto de Sistemas Robóticos, DCC/UFMG, segundo semestre de 2026.

Este TP traz dois comportamentos de navegação para o robô HeRo no simulador do
kit psr-hero-kit. O `avoid` desvia de obstáculos apenas com o lidar. O `goto`
leva o robô até as metas que o simulador publica, a partir da odometria.

## Estrutura

- `exercicios/avoid/avoid.py`, nó de desvio de obstáculos.
- `exercicios/goto/goto.py`, nó de navegação até metas.
- `exercicios/avoid/*.csv` e `exercicios/goto/*.csv`, registros dos dois ensaios
  de cada exercício, um arquivo por configuração de parâmetros.
- `gerar_graficos.py`, script que produz as figuras e as tabelas de métricas.
- `graficos/`, figuras geradas, uma por arquivo.
- `video-avoid.mp4` e `video-goto.mp4`, vídeos curtos de cada comportamento no
  simulador.

## Como reproduzir os ensaios

Os ensaios rodam dentro do container do kit psr-hero-kit. A partir da raiz do
kit, os comandos principais são os seguintes.

```bash
./run.sh build            # prepara o ambiente, uma vez
./run.sh run avoid        # abre o simulador e roda exercicios/avoid/avoid.py
./run.sh run goto         # roda exercicios/goto/goto.py
```

Cada nó grava o ensaio em um CSV na própria pasta, com o tempo, a pose e o
comando. O `avoid` grava também a menor distância do lidar, e o `goto` grava a
meta atual. O ensaio do `avoid` dura cinco minutos e o do `goto` dura três
minutos.

As duas configurações de parâmetros ficam no topo de cada script.

- No `avoid`, a Config. 1 tem `K_V=0.6, K_W=8.0, D_SLOW=0.25` e a Config. 2,
  ativa no código, tem `K_V=1.0, K_W=4.0, D_SLOW=0.145`.
- No `goto`, a Config. 1 tem `K_RHO=1.0, K_ALPHA=4.0` e a Config. 2, ativa no
  código, tem `K_RHO=3.0, K_ALPHA=6.0`.

Para repetir a Config. 1, troque os valores ativos pelos da Config. 1 no topo do
script e rode de novo.

## Como gerar os gráficos

Fora do container, com numpy e matplotlib, a partir desta pasta.

```bash
MPLBACKEND=Agg python3 gerar_graficos.py
```

O script lê os CSV de `exercicios/`, escreve as figuras em `graficos/` e grava o
`resumo.md` com as tabelas de métricas.

## Vídeos

Os arquivos `video-avoid.mp4` e `video-goto.mp4` mostram cada comportamento no
simulador.

## Equipe

Vinicius Trindade Dias Abel, Guilherme Buxbaum Marinho Guerra, Thiago Henrique
Silva de Almeida, Rafael Santos Oliveira.
