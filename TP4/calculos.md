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
- **Troca do motor (07/11/2026):** o SG90 de giro contínuo não estava
  funcionando corretamente e acabou parando de funcionar. Foi substituído por um
  servo **TowerPro MG90S**, no mesmo header (sinal no D10), com a mesma roda,
  encoder e transmissão. Por isso N (seção 1), D e k (seção 2) continuam
  valendo. O neutro e a banda morta da seção 4 e os pontos preliminares da
  seção 5 foram medidos com o SG90 e serão medidos novamente com o MG90S.

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
usado é **k/Δt = 0,4746 / 0,050 = 9,5 mm/s**.

Conferido com o motor girando, no log da curva aberta (1259 amostras, `dt_us` em
cada linha): média 50,000 ms, desvio 9,2 µs, faixa de 49860 a 50140 µs (±0,3 %).
O Δt real não se afasta do nominal de forma que mude o degrau.

Numa velocidade
baixa, por exemplo 30 mm/s com Δt = 50 ms, a janela tem só cerca de 3,2 ticks
(1,5 mm / 0,4746), e um tick a mais ou a menos muda a leitura em cerca de 32 %.
É daí que vem a necessidade do filtro.

## 4. Neutro do servo (busca na bancada)

### 4.1 Servo atual (motor trocado)

O servo foi trocado depois da primeira busca, porque as duas corridas com o
servo anterior não concordavam entre si. Os dados abaixo são do servo novo.
Comando `N 1300 1700`: cada pulso por 5 s, descartando o primeiro 1 s, ticks
contados nos 4 s seguintes, passos de 10 µs. "Parada" = no máximo 2 ticks em
4 s. `v = k · Δticks / Δt`, com k = 0,4746 mm/tick. A banda morta é a da
bancada inteira (servo, transmissão e roda), vista pelo encoder.

| Pulso (µs) | Ticks em 4 s | v (mm/s) | Estado |
|---|---|---|---|
| 1300 | -1182 | -142,01 | gira − |
| 1310 | -1135 | -136,36 | gira − |
| 1320 | -1083 | -130,12 | gira − |
| 1330 | -1044 | -125,43 | gira − |
| 1340 | -1017 | -122,19 | gira − |
| 1350 | -981 | -117,86 | gira − |
| 1360 | -958 | -115,10 | gira − |
| 1370 | -924 | -111,01 | gira − |
| 1380 | -904 | -108,61 | gira − |
| 1390 | -858 | -103,08 | gira − |
| 1400 | -840 | -100,92 | gira − |
| 1410 | -810 | -97,32 | gira − |
| 1420 | -771 | -92,63 | gira − |
| 1430 | -739 | -88,79 | gira − |
| 1440 | -705 | -84,70 | gira − |
| 1450 | -690 | -82,90 | gira − |
| 1460 | -626 | -75,21 | gira − |
| 1470 | 0 | 0,00 | parada |
| 1480 | 0 | 0,00 | parada |
| 1490 | 0 | 0,00 | parada |
| 1500 | 0 | 0,00 | parada |
| 1510 | 0 | 0,00 | parada |
| 1520 | 0 | 0,00 | parada |
| 1530 | 667 | 80,14 | gira + |
| 1540 | 690 | 82,90 | gira + |
| 1550 | 742 | 89,15 | gira + |
| 1560 | 761 | 91,43 | gira + |
| 1570 | 795 | 95,52 | gira + |
| 1580 | 839 | 100,80 | gira + |
| 1590 | 865 | 103,93 | gira + |
| 1600 | 904 | 108,61 | gira + |
| 1610 | 948 | 113,90 | gira + |
| 1620 | 990 | 118,94 | gira + |
| 1630 | 1013 | 121,71 | gira + |
| 1640 | 1069 | 128,44 | gira + |
| 1650 | 1108 | 133,12 | gira + |
| 1660 | 1140 | 136,97 | gira + |
| 1670 | 1160 | 139,37 | gira + |
| 1680 | 1192 | 143,21 | gira + |
| 1690 | 1214 | 145,86 | gira + |
| 1700 | 1267 | 152,22 | gira + |

- Último pulso com a roda girando para trás: 1460 µs (−75,2 mm/s)
- Pulsos com a roda parada: de 1470 a 1520 µs
- Primeiro pulso com a roda girando para a frente: 1530 µs (+80,1 mm/s)
- **Neutro = (1470 + 1520) / 2 = 1495 µs**
- Banda morta ≈ 1465 a 1525 µs, ≈ 60 µs (resolução de 10 µs)

Observações:
- A roda pula de 0 para ≈ ±75 a 80 mm/s ao sair da banda morta, nos dois
  sentidos. Não existe velocidade sustentável entre 0 e ≈ 75 mm/s com pulso
  fixo, o que limita o primeiro patamar da malha fechada.
- Fora da banda a curva é aproximadamente linear: de 1530 a 1700 µs, a
  velocidade sobe de 80 a 152 mm/s (≈ 0,42 mm/s por µs), e de 1460 a 1300 µs
  vai de −75 a −142 mm/s. Ainda não saturou em ±200 µs do neutro; a
  saturação sai da varredura `A`.
- Simetria: a 100 µs do neutro, +108,6 mm/s (1600 µs) e −100,9 mm/s (1400 µs),
  cerca de 7 % de diferença. A 200 µs, +152,2 e −142,0 mm/s (também ≈ 7 %).
- O neutro mudou 10 µs em relação ao servo anterior (1485 µs), com a mesma
  largura de banda morta (≈ 60 µs).
- **Repetição (2ª corrida, mesmo comando):** a faixa parada saiu igual, de 1470
  a 1520 µs, e o neutro de novo em **1495 µs** (banda morta de 60 µs). O neutro
  se repete. Nas velocidades houve diferença entre as corridas: em média
  −6,5 mm/s no lado negativo (a 2ª corrida foi mais rápida para trás) e
  +2,3 mm/s no positivo, com diferença máxima de 8,8 mm/s (≈ 8 %) em um mesmo
  pulso. Extremos da 2ª corrida: −146,1 mm/s em 1300 µs e +153,9 mm/s em
  1700 µs. A velocidade para um dado pulso varia cerca de ±5 mm/s de uma
  corrida para outra, e isso é a razão de fechar a malha.

### 4.2 Servo anterior (descartado, só registro)

Firmware `firmware/controle`, comando `N 1400 1600`, com o **servo anterior**: aplica cada pulso por 5 s,
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

Comando `A 50 3` com o servo atual e neutro de 1495 µs: de 1000 a 2000 µs em
passos de 50 µs, 3 s por pulso. Registro em `dados/malha_aberta_A50_3.csv`
(1259 amostras de 50 ms). Para cada pulso, a velocidade em regime é
`v = k · Δticks / Δt` entre 1,5 s e 3 s do pulso (descartando o transitório),
com k = 0,4746 mm/tick. Reproduzir: `python3 scripts/curva_aberta.py`
(gera `dados/curva_aberta.csv` e, com matplotlib, `figuras/curva_aberta.png`).

| Pulso (µs) | v em regime (mm/s) | desvio da v bruta (mm/s) |
|---|---|---|
| 1000 | -258,0 | 12,9 |
| 1050 | -240,6 | 11,2 |
| 1100 | -221,9 | 12,6 |
| 1150 | -205,9 | 11,2 |
| 1200 | -184,3 | 11,1 |
| 1250 | -166,6 | 13,6 |
| 1300 | -147,0 | 8,2 |
| 1350 | -121,8 | 13,6 |
| 1400 | -104,4 | 5,6 |
| 1450 | -81,8 | 11,0 |
| 1500 | 0,0 | 0,0 |
| 1550 | 91,6 | 10,7 |
| 1600 | 111,6 | 10,5 |
| 1650 | 132,2 | 12,8 |
| 1700 | 154,2 | 10,7 |
| 1750 | 174,1 | 15,7 |
| 1800 | 191,8 | 13,7 |
| 1850 | 215,4 | 13,1 |
| 1900 | 233,4 | 15,5 |
| 1950 | 253,7 | 13,5 |
| 2000 | 272,6 | 13,3 |

- **Extremos:** +272,6 mm/s em 2000 µs e −258,0 mm/s em 1000 µs. A assimetria
  entre os sentidos é de 14,6 mm/s, cerca de 5,4 % do maior.
- **Banda morta:** 1500 µs dá 0 ticks, e 1450 e 1550 µs já giram (−81,8 e
  +91,6 mm/s). Isso concorda com a busca da seção 4: parada de 1470 a 1520 µs,
  e salto de 0 para ≈ ±80 mm/s ao sair dela.
- **Saturação:** não há patamar plano. A curva segue quase linear até as pontas,
  com inclinação de ≈ 0,38 a 0,40 mm/s por µs no lado positivo (de +91,6 a
  +272,6 mm/s entre 1550 e 2000 µs) e um pouco menor perto de 1000 µs (de
  −258,0 a −240,6 mm/s nos primeiros 50 µs, ≈ 0,35 mm/s por µs). Dentro de
  1000 a 2000 µs o servo não satura de forma clara.
- **Ruído:** a velocidade bruta tem desvio de 5 a 16 mm/s em regime, com passo
  de quantização de 9,5 mm/s. É o que o filtro precisa tratar.
- **Faixa útil para a malha:** de ≈ 80 mm/s (saída da banda morta) até
  ≈ 250 mm/s. Os patamares do ensaio 1 devem ficar nessa faixa, por exemplo
  100, 130, 160, 190 e 220 mm/s (ajustar com `q`).
- Os pontos preliminares anteriores (+273 e −258 mm/s do teste de sanidade) foram
  substituídos por esta varredura.

## Pendências

- [x] Medir D e calcular k
- [x] Registrar o método de D (paquímetro, 43,40 mm)
- [x] Δt do código (50 ms) e degrau k/Δt (9,5 mm/s)
- [x] Neutro do servo atual: 1495 µs, confirmado em 2 corridas (seção 4.1)
- [x] Curva pulso × velocidade em malha aberta (mm/s): seção 5 (falta só a figura, que precisa de matplotlib)
- [ ] Comparação de dois filtros no mesmo degrau
- [ ] Sintonia do PID (tabela de tentativas)
- [ ] Ensaios: cinco patamares, inversão de sentido, distância

## Confirmações opcionais

- [ ] Se a roda ou a engrenagem do encoder foram mexidas ao trocar o motor, girar
  uma volta à mão (`z` antes, para zerar) e conferir se dá ≈ 287 ticks. Se der
  perto de 285 a 290, o N = 287,3 continua valendo; se desviar muito, repetir a
  medida de N.
- [ ] Anotar a data e as condições de cada ensaio (em especial a alimentação do
  servo: USB, fonte externa ou pilhas), porque o relatório declara a condição de
  cada ensaio (neutro, k, Δt, ganhos e parâmetro do filtro).
