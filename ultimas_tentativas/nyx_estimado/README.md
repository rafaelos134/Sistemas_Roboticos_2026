# Nyx estimado — engenharia reversa do rouba-bandeira do Grupo 3 · Nyx

Reconstrução do comportamento do time **Nyx** no rouba-bandeira, feita **só a partir dos trajetos**
gravados ao vivo no torneio (33 partidas, `../gravacoes_ao_vivo/ctf`). O código deles não foi visto, e os
logs de texto deles estão vazios. O resultado é um nó no formato do kit (`nyx_estimado.py`), para usar
como adversário de teste e de treino.

Tudo está no **referencial do time do Nyx**: a base dele fica em (−0,62; 0) e ele ataca +x.
Constantes do juiz (kit): segurar a < 0,10 m (só no próprio campo), pegar a bandeira a < 0,07 m,
4 s preso e depois 1,5 s imune.

## O que foi encontrado

### Defensor (bem determinado)
| Observação | Número |
|---|---|
| Fica **em cima da própria bandeira** | mediana a 4 mm da base, 63 % do tempo de jogo a < 5 cm |
| Parado, **vira de frente para o atacante inimigo** | ângulo coincide com a direção do atacante (−14° ≈ −13°) |
| **Só sai quando roubam a bandeira** | 381 das 397 saídas |
| Persegue **direto quem carrega a bandeira**, colado | a ~7 cm; segura em 71 % das saídas |
| **Desiste no meio do campo** (x ≈ −0,03) e volta de frente para a base | ex.: partida 678, 44–46 s |
| **Falha:** se quem carrega foi segurado *e* pegou a bandeira ao mesmo tempo, o defensor continua "perseguindo" e **empurra** o robô preso até a base dele | partida 650, 93–101 s: empurrou 1,1 m e deu o ponto ao adversário |

### Atacante (aproximado)
| Observação | Número |
|---|---|
| Vai **direto** para a bandeira inimiga em velocidade máxima | 0,20 m/s, giro ≈ 0 |
| **Nunca chega a menos de ~0,11 m do defensor inimigo** (o juiz segura a 0,10) | p1 = 0,108, p5 = 0,126, mediana 0,140 m |
| Quando o defensor inimigo cobre a bandeira, **orbita** em volta dele | raio ≈ 0,165 m, v ≈ 0,2 m/s, ω ≈ 1,2 rad/s (= v/r), bandeira a 90° do lado |
| Atrás da bandeira **gira no lugar** na velocidade máxima (4 rad/s) e testa outro ângulo; fica parado 61 % do tempo perto da bandeira | |
| Se foi **segurado perto da bandeira**, ao ser solto vai para um **ponto de espera** (−0,10; ±0,33), **do lado oposto** ao que estava, e espera **~10 s** | 26 de 26 vezes; lado oposto em 92 % |
| Com a bandeira: volta para casa, mas é segurado no caminho | 50 pegadas, **0 pontos**, 70 % presos na volta |

### Resultado real do Nyx no torneio (33 partidas)
0 pontos feitos, 87 sofridos (13 empates, 20 derrotas). **Contra o R2D2: 5 × 0 a 0.**
Quem marca nele (Claude 6 × 0, Tech 6 × 0, RTech 5 × 0) aproveita o defensor parado em cima da bandeira:
o atacante encosta, é segurado ao lado da bandeira e, ao ser solto, **pega a bandeira nos 1,5 s de
imunidade** (aparece nas gravações como "preso + carregando"). Depois disso o defensor do Nyx persegue
e às vezes empurra o carregador até a base adversária.

## Validação (`nyx_validar.py`, resultado em `validacao.txt`)

1. **Malha aberta**: o nó recebe cada estado real gravado e o comando dele é comparado com o movimento
   real (erro médio em unidades de V_MAX / W_MAX; "média" = prever sempre o comando médio).

   | | nyx_estimado | prever a média | imitação por rede (`bc_grupo_3_nyx_*`) |
   |---|---|---|---|
   | defensor | **0,120** | 0,253 | 0,053 |
   | atacante | 0,518 (0,472 só com o robô em movimento) | 0,341 | 0,065 |

   O **defensor** é bem explicado pelas regras. O **atacante** acerta a ida reta e a órbita, mas a
   sequência de estados (quando rodar, girar no lugar, desistir e esperar) ainda se desencontra do real
   — ele **não** está fiel passo a passo. Parte do erro vem de momentos em que o robô real está
   empurrando a parede atrás da bandeira (o movimento deduzido é 0, mas o comando não).
2. **Malha fechada** no psr_sim contra o nosso `ctf_hibrido_v4`, 4 partidas de 3 min: **0 × 0 em todas**,
   igual às 5 partidas reais R2D2 × Nyx. Defensor na base 90 % (real 63 %: contra o v4 a bandeira é
   menos roubada), atacante a ≥ 0,10 m do defensor (real ≥ 0,108).

**Para treino**: use o defensor deste nó e, se quiser um atacante mais fiel, a rede `bc_grupo_3_nyx_att`
(`../rl/runs/`). O defensor de regras é melhor que a rede em situações que não apareceram nas gravações,
porque segue regras fixas.

## Como usar

```bash
# no container, pasta rl do kit (ver ../rl/README.md)
cp ultimas_tentativas/nyx_estimado/nyx_estimado.py exports/ctf/
~/ros2_ws/.venv/bin/python eval_ctf.py --blue exports/ctf/ctf_hibrido_real1.py --red exports/ctf/nyx_estimado.py --minutes 3 --viewer
~/ros2_ws/.venv/bin/python nyx_validar.py            # precisa de data/ctf_ao_vivo (gravações do ctf)
# no torneio/kit:  ./run.sh run ctf blue:=nyx_estimado.py red:=...
```

`analise/` tem os scripts da investigação (numpy + pandas), na ordem em que foram feitos:
`a1` posição do defensor, `a2` saídas do defensor, `a3`/`a5`/`b1` séries no tempo, `a4` atacante por
zona, `a6` distância ao defensor inimigo e volta com a bandeira, `a7`–`a9` ponto de espera e o
gatilho (ser segurado).
