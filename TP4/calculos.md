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
- Servo SG90 de giro contínuo no header Servo1 do shield de motores ("Motor
  Driver V2.0", provável clone do L293D), com sinal no **pino D10**. O enunciado
  indica D9. O teste de sanidade (`firmware/teste_servo`) confirmou que o servo
  gira com o sinal no D10: pulso de 2000 µs dá ticks crescentes e 1000 µs dá
  ticks decrescentes, então A e B nos pinos 2 e 3 estão no sentido certo.
  (O D9 não foi testado.)

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

D foi medido com **paquímetro**, na roda montada, na banda que toca o chão:

**D = 43,40 mm** (resolução assumida do paquímetro: ±0,05 mm)

(Uma medida anterior, pela volta marcada no papel, deu 46 mm e foi descartada
por ter incerteza bem maior, cerca de ±1 mm no perímetro.)

- Perímetro: π · 43,40 = **136,35 mm**
- **k = 136,35 / 287,3 = 0,4746 mm/tick**
- 1 / k = **2,107 tick/mm**
- Incerteza de k vinda de N: 0,4746 × 0,57 / 287,3 = ± 0,0009 mm/tick
- Incerteza de k vinda de D: 0,4746 × 0,05 / 43,40 = ± 0,0005 mm/tick
- Combinando em quadratura: **k = 0,4746 ± 0,0011 mm/tick** (0,2 %). Agora N
  domina a incerteza, não mais D.

Conferência com a bancada de referência: π × 49,2 / 288 = 0,537 mm/tick. O k da
montagem é cerca de 12 % menor (43,40 / 49,20 = 0,882, ou seja, a roda tem
11,8 % menos diâmetro). A diferença vem do diâmetro da roda, não de N, que
bate com 288 dentro de 1,3 erro padrão.

## 3. Degrau de quantização da velocidade

A velocidade anda em degraus de `k / Δt`, porque em uma janela só cabem ticks
inteiros. Com k = 0,4746 mm/tick:

| Δt | Degrau k/Δt |
|---|---|
| 20 ms | 23,7 mm/s |
| 50 ms | 9,5 mm/s |
| 100 ms | 4,7 mm/s |

**Δt do código:** período nominal de 50 ms, agendado por `micros()`, e o Δt da
conta é o intervalo medido entre as leituras. Numa amostra de 33 linhas do log
(firmware de controle, roda parada), `dt_us` ficou em 50000,1 ± 4,8 µs (faixa de
49992 a 50008 µs). Portanto Δt = 50,000 ms, com desvio de 0,005 ms, e o degrau
usado é **k/Δt = 0,4746 / 0,050 = 9,5 mm/s**. (Refazer a média e o desvio com o
log completo dos ensaios, que traz `dt_us` em cada linha.)

Numa velocidade
baixa, por exemplo 30 mm/s com Δt = 50 ms, a janela tem só cerca de 3,2 ticks
(1,5 mm / 0,4746), e um tick a mais ou a menos muda a leitura em cerca de 32 %.
É daí que vem a necessidade do filtro.

## 4. Neutro do servo (busca na bancada)

Firmware `firmware/controle`, comando `N 1400 1600`: aplica cada pulso por 5 s,
descarta o primeiro 1 s (transitório) e conta os ticks nos 4 s seguintes.
Passos de 10 µs. Critério de "parada": no máximo 2 ticks em 4 s. A velocidade é
`v = k · Δticks / Δt` com k = 0,4746 mm/tick. A banda morta é a da bancada
inteira (servo, transmissão e roda), vista pelo encoder, e não a do servo solto.

| Pulso (µs) | Ticks em 4 s | v (mm/s) | Estado |
|---|---|---|---|
| 1400 | -784 | -94,19 | gira − |
| 1410 | -766 | -92,03 | gira − |
| 1420 | -725 | -87,11 | gira − |
| 1430 | -677 | -81,34 | gira − |
| 1440 | -649 | -77,97 | gira − |
| 1450 | -86 | -10,33 | gira − |
| 1460 | 0 | 0,00 | parada |
| 1470 | 0 | 0,00 | parada |
| 1480 | 0 | 0,00 | parada |
| 1490 | 0 | 0,00 | parada |
| 1500 | 0 | 0,00 | parada |
| 1510 | 0 | 0,00 | parada |
| 1520 | 589 | 70,77 | gira + |
| 1530 | 622 | 74,73 | gira + |
| 1540 | 669 | 80,38 | gira + |
| 1550 | 707 | 84,94 | gira + |
| 1560 | 747 | 89,75 | gira + |
| 1570 | 777 | 93,35 | gira + |
| 1580 | 817 | 98,16 | gira + |
| 1590 | 851 | 102,24 | gira + |
| 1600 | 883 | 106,09 | gira + |

- Último pulso com a roda girando para trás: 1450 µs (−10,3 mm/s)
- Pulsos com a roda parada: de 1460 a 1510 µs
- Primeiro pulso com a roda girando para a frente: 1520 µs (+70,8 mm/s)
- **Neutro = ponto médio da faixa parada = (1460 + 1510) / 2 = 1485 µs**
- Banda morta ≈ 1455 a 1515 µs, ou seja, ≈ 60 µs (resolução de 10 µs)

Observações:
- O início do movimento é abrupto: a roda pula de 0 para ≈ +71 mm/s entre 1510
  e 1520 µs. Do lado negativo, vai de 0 a −10 mm/s em 1450 µs e a −78 mm/s em
  1440 µs. Há uma velocidade mínima sustentável, em torno de 70 mm/s para a
  frente, que o PID não consegue contornar com um pulso fixo.
- A banda é simétrica em torno de 1485 µs (30 µs para cada lado), mas a
  velocidade logo depois da borda não é: −10 a −78 mm/s embaixo (1450 e 1440 µs)
  e +71 mm/s em cima (1520 µs).
- **Pendência de consistência:** uma primeira corrida da mesma busca, de 1400 a
  1600 µs, deu a roda parada de 1450 a 1600 µs, sem girar para a frente, e
  1440 µs dava −30 mm/s (aqui, −78 mm/s). As duas corridas discordam do lado de
  cima. Registrar o que mudou entre elas (alimentação, aquecimento, carga) e
  repetir a busca para saber a repetibilidade do neutro.

## 5. Curva pulso × velocidade em malha aberta

Comando `A [passo] [s]`: varre de 1000 a 2000 µs, cada pulso por 3 s. A
velocidade em regime vem do log, descartando o transitório (por exemplo, o
primeiro 1,5 s de cada pulso), em mm/s: `v = k · Δticks / Δt` com k = 0,4746.

Pontos preliminares (teste de sanidade, ticks lidos de 0,5 em 0,5 s, em regime):

| Pulso (µs) | Δticks / 0,5 s | Velocidade |
|---|---|---|
| 2000 | ≈ 288 | ≈ 576 ticks/s = **+273 mm/s** |
| 1000 | ≈ −272 | ≈ −544 ticks/s = **−258 mm/s** |

Já mostra uma assimetria de cerca de 6 % entre os sentidos. A curva completa
(banda morta, saturação nas pontas) sai da varredura.

_Registrar aqui: tabela pulso × velocidade, banda morta, saturação, assimetria._

## Pendências

- [x] Medir D e calcular k
- [x] Registrar o método de D (paquímetro, 43,40 mm)
- [x] Δt do código (50 ms) e degrau k/Δt (9,5 mm/s)
- [~] Neutro do servo: 1485 µs medido (seção 4); falta repetir a busca para confirmar
- [ ] Curva pulso × velocidade em malha aberta (mm/s): seção 5
- [ ] Comparação de dois filtros no mesmo degrau
- [ ] Sintonia do PID (tabela de tentativas)
- [ ] Ensaios: cinco patamares, inversão de sentido, distância
