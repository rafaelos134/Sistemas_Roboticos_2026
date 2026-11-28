# Trabalhos: Projeto de Sistemas Robóticos

UFMG, DCC, 2º semestre de 2026, Prof. Paulo Rezeck. Turmas TE3, TL19 e P25.

Disciplina prática, baseada em laboratório e projeto integrador, que dá um panorama do projeto de sistemas robóticos: mecânica e fabricação digital, eletrônica e PCB, sistemas embarcados, sensores e filtragem, controle e odometria, e integração com ROS/ROS2. Os módulos rendem cinco relatórios e culminam em um projeto final coletivo (futebol de robôs, sumô ou pega-pega).


## Estrutura

| Pasta            | Relatório | Módulo                          |
|------------------|-----------|---------------------------------|
| `TP1/`           | R1        | Mecânica e fabricação digital   |
| `TP2/`           | R2        | Eletrônica e PCB                |
| `TP3/`           | R3        | Embarcado, sensores e filtragem |
| `TP4/`           | R4        | Controle e odometria            |
| `TP5/`           | R5        | ROS, navegação e integração     |
| `projeto_final/` | Final     | Sumô, futebol e rouba-bandeira: **entrega** (clássico, RL e treino) |
| `projeto-final/` | —         | Pasta lida pelo servidor do torneio (versão em jogo) |

```
trabalhos/
├── TP1/
│   ├── relatorio/        # Relatorio_1.pdf
│   ├── modelo_3d/        # caixa_planetaria1.FCStd
│   └── imagens/
│       ├── cotadas/
│       └── parametrizacao/
├── TP2/                
├── TP3/
├── TP4/
├── TP5/
├── projeto_final/        # ENTREGA do projeto final (ver projeto_final/README.md)
│   ├── classico/         # sumo.py, soccer.py, ctf.py: máquinas de estados
│   ├── rl/               # sumo.py, soccer.py, ctf.py: políticas PPO finais
│   └── treinamento/      # código de treino, checkpoints e scripts para reproduzir as versões de rl/
└── projeto-final/        # o que o servidor do torneio executa (sumo.py, soccer.py, ctf.py)
```

### `projeto_final/` × `projeto-final/`

As duas pastas têm nomes quase iguais, mas papéis diferentes:

- **`projeto_final/` (com sublinhado) é a entrega para avaliação.** Tem as duas famílias de solução, separadas:
  `classico/` (máquinas de estados para os três jogos) e `rl/`
  (as versões finais de aprendizado por reforço), além de `treinamento/`, com o código de treino, os checkpoints
  finais e um script por jogo para reproduzir cada rede. O `README.md` dentro dela explica cada arquivo e como
  rodar, treinar e avaliar.
- **`projeto-final/` (com hífen) é a pasta de implantação do torneio.** O servidor da disciplina busca, a cada
  5 minutos, os arquivos `sumo.py`, `soccer.py` e `ctf.py` desta pasta no último *commit* da `main`, e é isso que
  joga as partidas de exibição. O conteúdo muda conforme a versão que queremos testar no torneio; hoje são as
  versões de RL (iguais às de `projeto_final/rl/`). O `ctf_robusto_backup.py` é uma alternativa do rouba-bandeira
  (atacante e defensor de RL) guardada para troca rápida; o servidor não o executa.

Para avaliar o trabalho, use `projeto_final/`.

As pastas vazias têm um `.gitkeep` só para o Git versionar o diretório.

## Tecnologias

**Tecnologias utilizadas:** Git, GitHub, VS Code, Arduino IDE, PlatformIO, FreeCAD, fatiador FDM (PrusaSlicer/Cura), EasyEDA, ROS/ROS2, RViz, Python, Matplotlib e Jupyter.

<p>
  <img src="https://skillicons.dev/icons?i=git,github,vscode,arduino,python,ros&perline=8" alt="Git, GitHub, VS Code, Arduino, Python, ROS" />
</p>

