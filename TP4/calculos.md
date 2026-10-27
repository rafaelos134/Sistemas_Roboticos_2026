# TP4: cálculos do encoder

Registro dos números medidos e das contas, na ordem em que foram feitos.
Símbolos do enunciado (Seção 3):

| Símbolo | Significado |
|---|---|
| D | diâmetro externo da roda, em mm, na banda de contato |
| N | ticks do encoder por volta completa da roda |
| k | avanço da roda por tick, em mm/tick: `k = π·D / N` |
| s | distância percorrida: `s = k · ticks` |
| v | velocidade linear: `v = Δs / Δt` |

Referência da bancada (só para conferência): D = 49,2 mm e N = 288 ticks
(48 pulsos do encoder e redução 1:6), resultando em k = 0,537 mm/tick.

## Montagem

- Encoder: TTC amarelo de mouse, mecânico, 3 pinos. Comum no GND, as pontas nos
  pinos 2 (A) e 3 (B), com pull-up interno. Sem ligação no 5 V.
- Servo SG90 de giro contínuo no header Servo1 do Motor Shield V2, ou seja,
  pino D10. O enunciado indica D9 (header Servo2).

## 1. Ticks por volta (N)

### 1.1 Tentativas descartadas

| Método | Resultado | Por que foi descartado |
|---|---|---|
| Código manda o motor girar até um número de ticks e a volta é conferida a olho | ≈ 366 | O servo passa do ponto depois de parar. O encoder perde transições com o motor girando. |
| À mão, com ISRs próprias (`isrA`/`isrB`), 10 voltas positivas e 3 negativas | 429,2 ± 5,5 (s = 20,0) | Espalhamento muito alto: faixa de 403 a 465. |

Leituras à mão com as ISRs próprias:
- positivo: 449, 465, 442, 418, 415, 418, 407, 463, 429, 403 (média 430,9, s = 22,6)
- negativo: −416, −427, −427 (média 423,3 em módulo, s = 6,4)

### 1.2 Medida adotada: à mão, com a biblioteca Encoder

Usou a biblioteca Encoder da verlab/hero_common (`lib/Encoder`), em X4, com 10
voltas inteiras em cada sentido, zerando o contador (`z` no serial) antes de
cada volta.

| Volta | Positivo | Negativo |
|---|---|---|
| 1 | 285 | −288 |
| 2 | 286 | −291 |
| 3 | 288 | −288 |
| 4 | 281 | −290 |
| 5 | 288 | −291 |
| 6 | 285 | −285 |
| 7 | 289 | −289 |
| 8 | 287 | −283 |
| 9 | 288 | −288 |
| 10 | 287 | −289 |

Estatística, com valores em módulo:

| Sentido | n | Média | Mediana | Desvio s | s/√n | Faixa |
|---|---|---|---|---|---|---|
| Positivo | 10 | 286,4 | 287,0 | 2,32 | 0,73 | 281 a 289 |
| Negativo | 10 | 288,2 | 288,5 | 2,53 | 0,80 | 283 a 291 |
| Todas | 20 | **287,3** | 288,0 | 2,54 | 0,57 | 281 a 291 |

Comparação dos sentidos (teste t de Welch):
- diferença das médias = 288,2 − 286,4 = 1,8 ticks
- erro padrão da diferença = √(0,73² + 0,80²) = 1,09 ticks
- t = 1,8 / 1,09 = 1,66, com cerca de 18 graus de liberdade, o que dá p ≈ 0,11

A diferença não é significativa a 5 %, então os dois sentidos são tratados como
iguais e se adota um N único.

**N adotado = 287,3 ± 0,6 ticks/volta.** O valor é compatível com a referência
de 288 ticks (48 × 6). A diferença de 0,7 tick fica dentro de 1,3 erro padrão.

Comparação entre os métodos: com a biblioteca, o desvio caiu de 20 para 2,5
ticks e a média passou a bater com o modelo da bancada. Os 429 das ISRs
próprias e os 366 da calibração com o motor ficam como registro de que a
contagem inicial não era confiável.

## 2. Constante k

`k = π · D / N`, com N = 287,3.

D foi medido duas vezes, na roda montada:

| Medida | Valor | D |
|---|---|---|
| 1ª: perímetro (π·D) | 14,45 cm = 144,5 mm | 144,5 / π = 46,0 mm |
| 2ª: diâmetro direto | 46 mm | 46 mm |

(Métodos: _registrar aqui qual instrumento foi usado em cada medida, por
exemplo volta marcada sobre papel ou paquímetro na banda de contato_.)

As duas medidas concordam (π × 46 = 144,5 mm).

- **D adotado = 46,0 mm** (o da referência é 49,2 mm, então este é 6,5 % menor)
- **k = 144,5 / 287,3 = 0,5030 mm/tick**
- 1 / k = **1,988 tick/mm**
- Incerteza de k vinda só de N: 0,5030 × 0,57 / 287,3 = ± 0,001 mm/tick.
  A incerteza do perímetro deve dominar. Com uma leitura de ± 1 mm na volta
  marcada, k varia ± 0,0035 mm/tick (0,7 %).

Conferência com a bancada de referência: π × 49,2 / 288 = 0,537 mm/tick. O k da
montagem é cerca de 6 % menor, e a diferença vem do diâmetro da roda, não de N.

## 3. Degrau de quantização da velocidade (pendente: Δt do código)

A velocidade anda em degraus de `k / Δt`, porque em uma janela só cabem ticks
inteiros. Com k = 0,5030 mm/tick:

| Δt | Degrau k/Δt |
|---|---|
| 20 ms | 25,1 mm/s |
| 50 ms | 10,1 mm/s |
| 100 ms | 5,0 mm/s |

Confirmar com o Δt que o código de controle realmente usar. Numa velocidade
baixa, por exemplo 30 mm/s com Δt = 50 ms, a janela tem só cerca de 3 ticks, e
um tick a mais ou a menos muda a leitura em 33 %. É daí que vem a necessidade
do filtro.

## Pendências

- [x] Medir D e calcular k
- [ ] Registrar o método de cada medida de D
- [ ] Δt do código e degrau k/Δt
- [ ] Neutro do servo (busca em passos de 10 µs em torno de 1500 µs)
- [ ] Curva pulso × velocidade em malha aberta (mm/s)
- [ ] Comparação de dois filtros no mesmo degrau
- [ ] Sintonia do PID (tabela de tentativas)
- [ ] Ensaios: cinco patamares, inversão de sentido, distância
