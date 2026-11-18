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
- **Histórico dos motores:** a bancada começou com um servo **MG90S**, que deu
  erro. Foi trocado por um **SG90**, que também deu erro, e depois por um
  **segundo MG90S, novo, que é o servo atual** (sinal no D10, mesma roda,
  encoder e transmissão em todos os casos). As seções 1 a 3 (N, D e k, e o Δt do
  firmware) foram feitas com o MG90S antigo e continuam valendo, porque dependem
  só da roda, do encoder e do firmware, e não do motor. O Δt da seção 3 também foi
  conferido com o motor novo girando. **Todas as medidas a partir da seção 4
  (neutro, curva aberta, filtros, PID e ensaios) são do MG90S novo.**

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

### 4.1 Servo atual (MG90S novo)

Os dados abaixo são do MG90S novo, o servo atual. Comando `N 1300 1700`: cada pulso por 5 s, descartando o primeiro 1 s, ticks
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
- O neutro mudou 10 µs em relação ao motor anterior (1485 µs, seção 4.2), com a mesma
  largura de banda morta (≈ 60 µs).
- **Repetição (2ª corrida, mesmo comando):** a faixa parada saiu igual, de 1470
  a 1520 µs, e o neutro de novo em **1495 µs** (banda morta de 60 µs). O neutro
  se repete. Nas velocidades houve diferença entre as corridas: em média
  −6,5 mm/s no lado negativo (a 2ª corrida foi mais rápida para trás) e
  +2,3 mm/s no positivo, com diferença máxima de 8,8 mm/s (≈ 8 %) em um mesmo
  pulso. Extremos da 2ª corrida: −146,1 mm/s em 1300 µs e +153,9 mm/s em
  1700 µs. A velocidade para um dado pulso varia cerca de ±5 mm/s de uma
  corrida para outra, e isso é a razão de fechar a malha.

### 4.2 Motor anterior (descartado, só registro)

Firmware `firmware/controle`, comando `N 1400 1600`, com o **motor anterior** (antes da última troca): aplica cada pulso por 5 s,
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

Comando `A 50 3` com o MG90S novo e neutro de 1495 µs: de 1000 a 2000 µs em
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
  ≈ 165 mm/s. O servo chega a 272 mm/s em malha aberta, mas a correção do PID é
  limitada a ±250 µs (pulso máximo de 1745 µs, ≈ 170 mm/s na curva), e a malha
  não passa disso (ver seção 8.1). Os patamares do ensaio 1 devem ficar nessa
  faixa, por exemplo 100, 115, 130, 145 e 160 mm/s (ajustar com `q`).
- Os pontos preliminares anteriores (+273 e −258 mm/s do teste de sanidade) foram
  substituídos por esta varredura.

## 6. Comparação de dois filtros no mesmo degrau

Degrau em malha aberta com o MG90S novo: `o 1700` a partir do repouso (comando
`python3 scripts/coleta.py filtros_degrau_o1700 "o 1700" --duracao 4`).
Registro em `dados/filtros_degrau_o1700.csv` (Δt = 50 ms). Os filtros são
refeitos a partir da velocidade bruta por `scripts/filtros.py`; o gráfico sai de
`scripts/grafico_filtros.py` (`figuras/filtros_degrau.png`). Regime: t ≥ 5 s, com
velocidade final de 152,4 mm/s.

- **Média móvel**, janela de n amostras (n × 50 ms): `y = (v[k] + ... + v[k−n+1]) / n`
- **Passa-baixa de primeira ordem**, parâmetro α: `y[k] = y[k−1] + α (v[k] − y[k−1])`

| Filtro | Parâmetro | Desvio em regime (mm/s) | Desvio de dv/dt (mm/s²) | t63 (s) | t90 (s) | Pico (mm/s) |
|---|---|---|---|---|---|---|
| Bruta | | 13,16 | 429,4 | 0,082 | 0,124 | 189,8 |
| Média móvel | n = 3 | 5,38 | 105,8 | 0,139 | 0,217 | 155,1 |
| Média móvel | **n = 5** | 4,49 | 80,9 | 0,210 | 0,321 | 155,7 |
| Média móvel | n = 8 | 3,44 | 50,2 | 0,316 | 0,438 | 154,2 |
| Média móvel | n = 10 | 2,85 | 39,8 | 0,386 | 0,546 | 152,8 |
| Passa-baixa | α = 0,15 | 2,49 | 42,0 | 0,408 | 0,892 | 151,5 |
| Passa-baixa | **α = 0,3** | 4,18 | 89,5 | 0,207 | 0,437 | 153,8 |
| Passa-baixa | α = 0,5 | 6,28 | 163,5 | 0,129 | 0,331 | 162,4 |

O gráfico do relatório compara a bruta com a média móvel n = 5 e o passa-baixa
α = 0,3, em negrito na tabela.

- **Ruído:** os dois cortam o desvio de 13,2 para ≈ 4,3 mm/s (cerca de 3 vezes) e
  o ruído de dv/dt de 429 para 81 a 90 mm/s², o que importa para o termo
  derivativo. A velocidade bruta tem picos de 190 mm/s em um regime de 152 mm/s.
- **Atraso:** os dois levam ≈ 0,21 s para chegar a 63 %. A média móvel assenta
  em 0,32 s (t90) e o passa-baixa em 0,44 s, porque a resposta ao degrau do
  passa-baixa tem uma cauda longa.
- **Troca ruído × atraso:** em ambos, mais suavização (n maior, α menor) baixa o
  ruído e aumenta o atraso. O atraso entra na malha fechada e limita o `kp`.
- **Escolha para o PID: média móvel com n = 5 (janela de 250 ms)**, com
  `f 0` no firmware. Justificativa:
  1. **Ruído:** desvio em regime de 4,5 mm/s, contra 13,2 mm/s da bruta (cerca de
     3 vezes menos), praticamente igual ao do passa-baixa α = 0,3 (4,2 mm/s).
  2. **Atraso:** o primeiro cruzamento de 63 % ocorre em 0,21 s, o mesmo do
     passa-baixa α = 0,3. Mas a média móvel assenta em 0,32 s (t90), contra
     0,44 s do passa-baixa, que tem cauda longa. Com o mesmo ruído, a média móvel
     entrega a velocidade correta mais cedo, e o atraso entra na malha
     fechada: quanto maior, menor o `kp` e o `ki` que a malha aguenta.
  3. **Derivativo e sobressinal:** o ruído de dv/dt é 81 mm/s² na média móvel,
     contra 90 mm/s² no passa-baixa e 429 mm/s² na bruta. O termo derivativo
     age na velocidade filtrada, então esse ruído iria direto para o pulso do
     servo. O pico do degrau é de 155,7 mm/s (cerca de 2 % acima dos 152,4 mm/s de
     regime, dentro do ruído), contra 189,8 mm/s da bruta.
  4. **Por que não mais suave:** n = 8 e n = 10 reduzem mais o ruído (3,4 e
     2,8 mm/s), mas o t63 sobe para 0,32 e 0,39 s e o t90 para 0,44 e 0,55 s. O
     passa-baixa α = 0,15 tem o menor ruído (2,5 mm/s), mas leva 0,89 s para
     chegar a 90 %. Esse atraso é alto demais para uma malha com patamares de 3 s.
  5. **Por que não mais rápido:** com n = 3 o ruído em regime ainda é de
     5,4 mm/s e o de dv/dt de 106 mm/s², e com α = 0,5 o pico passa de 162 mm/s
     (sobressinal de 6 % sobre os 152 mm/s de regime). O n = 5 é o ponto em que
     o ruído já é baixo e o atraso ainda é de uma ordem curta.
  6. **Limite da escolha:** a decisão vem de uma resposta em malha aberta. Se a
     malha fechada oscilar com a média móvel, comparar com o passa-baixa na mesma
     sintonia (`f 1`) e registrar na tabela de tentativas do PID.
- A janela de 5 amostras de 50 ms atrasa a velocidade em cerca de
  (n − 1) / 2 · Δt = 0,1 s. Somado ao t63 de 0,08 s da velocidade bruta, dá
  ≈ 0,18 s, próximo do t63 medido de 0,21 s.

## 7. Sintonia do PID

Filtro: média móvel n = 5 (`f 0`). Neutro 1495 µs, k = 0,4746 mm/tick, Δt = 50 ms.
Cada tentativa é um degrau de 0 a 150 mm/s durante 4 s, a partir do repouso:
`python3 scripts/coleta.py pid_tNN --pre "g kp ki kd" "s 150" --duracao 4`. Os
registros ficam em `dados/pid_tNN.csv`. Ganhos em µs/(mm/s) para `kp`,
µs/(mm/s)/s para `ki` e µs/(mm/s²) para `kd`. Regime: t ≥ 2,5 s.

| Tentativa | kp | ki | kd | O que o gráfico fez | Decisão |
|---|---|---|---|---|---|
| T01 (`pid_t01`) | 1 | 0 | 0 | Sem oscilação. v sobe em ≈ 0,25 s (63 %) e fica em **87,6 mm/s**, com erro de regime de 62,4 mm/s (42 % do setpoint). Pulso em 1557,6 µs (± 2,5 µs): correção de +62,6 µs, igual a kp · e. Pico de 93 mm/s em 0,45 s, dentro do ruído. | Subir kp para ver até onde o erro cai e onde a malha começa a oscilar. |
| T02 (`pid_t02`) | 3 | 0 | 0 | Sem oscilação. v chega a 140,5 mm/s em 0,35 s e se acomoda em **110,3 mm/s** (erro de regime 39,7 mm/s, 26 %). Pulso em 1614 µs, com variação de 1598 a 1626 µs (desvio 7,8 µs). O primeiro pulso satura em 1745 µs (correção de +250 µs), e por isso o pico passa do valor final. | Subir kp mais, procurando a oscilação. |
| T03 (`pid_t03`) | 6 | 0 | 0 | Começa a oscilar: pulso em 1651,6 µs, variando de 1598 a 1700 µs (desvio 27,7 µs), com período de ≈ 0,35 s na autocorrelação (0,61). v em **123,9 mm/s**, erro de regime 26,1 mm/s (17 %), com desvio de v de 4,6 mm/s (2,6 na T02). A oscilação é irregular, ruído de quantização amplificado pelo ganho mais o atraso do filtro, e não um ciclo limpo. | kp ≈ 6 é o limite prático (Ku ≈ 6, Tu ≈ 0,35 s). Voltar para kp ≈ 3 e subir ki. |
| T04 (`pid_t04`) | 3 | 3 | 0 | Sem oscilação. v sobe a 137 mm/s em 0,3 s e depois vai se aproximando devagar de 150 mm/s: nos últimos 1,5 s a média é **145,1 mm/s** (erro 4,9 mm/s, 3 %) e ainda sobe, o integrador não terminou em 4 s. Sem sobressinal (pico 151,9 mm/s em 3,3 s, no ruído). Pulso em 1701 µs (desvio 11,4 µs). | ki baixo demais para assentar em 4 s. Subir ki. |
| T05 (`pid_t05`) | 3 | 8 | 0 | Erro de regime **zero**: média de 150,0 mm/s nos últimos 1,5 s, pulso em 1709 µs (desvio 12,6 µs), que bate com a curva aberta (1700 µs ≈ 154 mm/s). Subida: 63 % em 0,25 s e 90 % em 0,35 s. Sobressinal de 7,6 % (pico 161,3 mm/s em 0,4 s), com o pulso saturado em 1745 µs (+250 µs) no início. A velocidade filtrada tem ruído de 3,8 mm/s em regime. | Candidata a ganhos finais. Testar ki = 5 para ver se baixa o sobressinal sem perder o regime. |

Leitura da T01: com só o termo proporcional, a correção `u = kp · e` precisa de
um erro de regime para existir. Para 150 mm/s a curva aberta pede ≈ 1660 µs
(correção de ≈ +165 µs), mas com kp = 1 a malha se acomoda em 1557 µs, onde a
curva dá ≈ 90 mm/s. O ganho estático da planta é de ≈ 0,4 mm/s por µs acima da
banda morta, então o erro de regime de um controle só P fica em torno de
`84 / (1 + 0,4 · kp)` mm/s (≈ 60 para kp = 1, ≈ 38 para kp = 3, ≈ 17 para
kp = 10). Só o termo integral zera esse erro.
Conferência com as tentativas seguintes: o erro de regime medido foi de 39,7 mm/s
com kp = 3 (previsto ≈ 38) e de 26,1 mm/s com kp = 6 (previsto ≈ 25), então o
modelo descreve bem a malha só P.

Ponto de partida do Ziegler–Nichols com os valores observados (Ku ≈ 6 µs/(mm/s),
Tu ≈ 0,35 s): kp = 0,6 Ku ≈ 3,6; ki = 1,2 Ku / Tu ≈ 21 µs/(mm/s)/s;
kd = 0,075 Ku Tu ≈ 0,16. Para um PI, o enunciado diz que costuma bastar um valor
bem ajustado, então a próxima tentativa parte de kp = 3 com ki bem menor que 21.

## 8. Ensaios em malha fechada

**Ganhos finais:** kp = 3 µs/(mm/s), ki = 8 µs/(mm/s)/s, kd = 0. Filtro: média
móvel n = 5. Neutro 1495 µs, k = 0,4746 mm/tick, Δt = 50 ms. Correção limitada a
±250 µs (pulso máximo 1745 µs, mínimo 1245 µs). A tentativa T06 (ki = 5) não foi
feita, e os ensaios usaram os ganhos da T05. Registros em `dados/`.

### 8.1 Ensaio 1: cinco patamares (primeira corrida, `ensaio1_patamares_saturado.csv`)

Patamares de 100, 130, 160, 190 e 220 mm/s, 3 s cada. Valores no último 1 s de
cada patamar.

| Patamar (mm/s) | v média | Erro | Sobressinal | t63 / t90 (s) | Pulso médio (µs) |
|---|---|---|---|---|---|
| 100 | 101,2 | −1,2 (−1,2 %) | 32,9 % (pico 132,9) | 0,20 / 0,25 | 1591 |
| 130 | 129,2 | 0,8 (0,6 %) | 5,1 % | 0,55 / 0,90 | 1658 |
| 160 | 159,6 | 0,4 (0,3 %) | 3,2 % | 0,75 / 1,60 | 1726 |
| 190 | 167,0 | **23,0 (12 %)** | não chegou | não chegou | **1745 (saturado)** |
| 220 | 167,6 | **52,4 (24 %)** | não chegou | não chegou | **1745 (saturado)** |

- Os três primeiros patamares ficam com erro de regime abaixo de 1,2 %.
- **Os patamares de 190 e 220 mm/s não são alcançáveis:** o pulso fica travado em
  1745 µs (neutro + 250 µs, o limite da correção) e a roda para em ≈ 167 mm/s.
  A curva aberta chega a 272 mm/s, mas só com pulsos até 2000 µs, e a correção do
  enunciado só deixa chegar a 1745 µs (≈ 170 mm/s na curva). A faixa útil da
  malha vai, portanto, de ≈ 80 a ≈ 165 mm/s, e não até 250 mm/s como estimei na
  seção 5.
- O sobressinal de 33 % no primeiro degrau (0 a 100 mm/s) vem do pulso saturado
  em 1745 µs no início e do integrador acumulando (pulso mínimo de 1491 µs ao
  voltar). Nos degraus seguintes ele cai para 3 a 5 %.
- A integral é zerada a cada troca de patamar. Por isso cada degrau recomeça o
  acúmulo e o t90 sobe de 0,25 s para 0,9 e 1,6 s. Não há carga antiga, mas o
  assentamento fica mais lento nos degraus pequenos.
- A corrida com 190 e 220 mm/s fica como evidência da saturação para a discussão.

#### Segunda corrida, dentro da faixa (`ensaio1_patamares.csv`)

Patamares de 100, 115, 130, 145 e 160 mm/s, 3 s cada, mesmos ganhos. Valores no
último 1 s de cada patamar.

| Patamar (mm/s) | v média | Erro | Desvio de v | Sobressinal | Pulso médio (µs) |
|---|---|---|---|---|---|
| 100 | 100,0 | 0,0 (0,0 %) | 2,7 | 31,0 % (pico 131,0) | 1592 |
| 115 | 114,4 | 0,6 (0,5 %) | 2,9 | 4,0 % | 1622 |
| 130 | 129,2 | 0,8 (0,6 %) | 2,9 | 3,7 % | 1650 |
| 145 | 144,7 | 0,3 (0,2 %) | 3,7 | 6,0 % | 1678 |
| 160 | 160,2 | −0,2 (−0,1 %) | 3,6 | 3,2 % | 1701 |

- **Erro de regime abaixo de 0,6 % em todos os patamares.** O integrador zera o
  erro, e o pulso médio de cada patamar acompanha a curva aberta (1592 µs para
  100 mm/s, 1701 µs para 160 mm/s).
- **Tempo de subida do primeiro degrau (0 a 100 mm/s):** 63 % em 0,25 s e 90 %
  em 0,30 s, com sobressinal de 31 % (pico 131,0 mm/s). O pulso satura em
  1745 µs nesse início e o integrador acumula.
- **Nos degraus seguintes** (de 15 mm/s cada), o sobressinal é de 3 a 6 % e o
  pulso volta a tocar o limite de 1745 µs nos de 145 e 160 mm/s. Para degraus tão
  pequenos, os tempos de 63 % e 90 % ficam entre 0,0 e 1,5 s e dependem do
  ruído de 3 mm/s da velocidade filtrada, então não são confiáveis.
- **A troca de patamar não deixa a integral para trás:** a integral é zerada em
  cada troca, e o erro volta a ficar abaixo de 1 % antes do fim dos 3 s em todos
  os patamares.

### 8.2 Ensaio 2: inversão de sentido (`ensaio2_inversao.csv`)

+120 mm/s por 4 s e depois −120 mm/s por 4 s.

- Regime positivo: 120,1 mm/s (pulso 1633 µs). Regime negativo: −120,6 mm/s
  (pulso 1369 µs). Erros de 0,1 e 0,5 mm/s.
- Na troca, o pulso vai direto para 1245 µs (−250 µs, saturado) por 0,25 s. A
  roda **passa por zero 0,15 s depois da troca**, atravessando a banda morta
  rápido, e chega a −117,7 mm/s em 0,25 s.
- Sobressinal de 31 % para o lado negativo (pico −157,6 mm/s em 0,3 s), por causa
  da saturação e do integrador zerado. Assenta em −121,5 mm/s em ≈ 0,55 s.
- **Sentido do encoder:** os ticks descem de +1001 para −20 quando a velocidade
  fica negativa, então o sinal do encoder concorda com o da velocidade, e não foi
  preciso trocar A e B.
- Deslocamento líquido: −9,5 mm (−20 ticks) depois de ≈ 475 mm para cada lado, ou
  cerca de 2 % do percurso de ida.

### 8.3 Ensaio 3: distância (primeira corrida, `ensaio3_distancia_v1_com_malha.csv`)

3 voltas a 120 mm/s: alvo π · D · 3 = **409,04 mm**. A roda chegou a 409,56 mm em
3,45 s (velocidade média de 120,6 mm/s antes disso).

- Problema: depois de atingir o alvo, o firmware mandava o setpoint para 0 e a
  malha continuava ativa. Por causa da banda morta, ela não consegue segurar
  velocidades entre −80 e +80 mm/s e fica caçando o zero, com o pulso indo de
  1245 a 1745 µs. A distância oscilou de 383 a 412 mm durante 1,5 s, e o valor
  impresso no fim (389,63 mm) não é uma medida de distância.
- **Correção no firmware:** ao atingir o alvo, o ensaio 3 passa a cortar o pulso
  direto no neutro (1495 µs), em malha aberta.

#### Segunda corrida, com o firmware corrigido (`ensaio3_distancia_v2.csv`)

- Velocidade média de 120,6 mm/s antes de chegar ao alvo.
- A roda chega a 410,5 mm em 3,45 s, quando o pulso é cortado para 1495 µs.
  Depois a roda ainda anda mais 3,8 mm e **para em 414,78 mm** (874 ticks), em
  cerca de 0,3 s, e fica parada. Sem caça ao zero.
- **Comparação com o comprimento da volta:** 3 · π · D = 409,04 mm (862 ticks).
  A diferença é de **+5,7 mm, ou +1,4 %** (874 contra 862 ticks).
- **Verificação à mão:** as três voltas conferiram "praticamente exatas" a olho,
  o que é coerente com a diferença de 1,4 % (≈ 6 mm em 409 mm).
- **De onde vêm os 5,7 mm:**
  - O alvo é testado a cada amostra de 50 ms, e a 120 mm/s a roda anda ≈ 6 mm
    por amostra, então o corte pode acontecer até 6 mm depois do alvo.
  - Depois do corte, a roda ainda desliza 3,8 mm até parar.
  - A incerteza de k (0,2 %) e de N (0,2 %) pesa menos (≈ 1 mm em 409 mm).
- Para reduzir o erro, bastaria antecipar o corte pela distância de frenagem
  (≈ 4 mm a 120 mm/s) ou testar o alvo com uma amostragem mais rápida.

## Pendências

- [x] Medir D e calcular k
- [x] Registrar o método de D (paquímetro, 43,40 mm)
- [x] Δt do código (50 ms) e degrau k/Δt (9,5 mm/s)
- [x] Neutro do servo atual: 1495 µs, confirmado em 2 corridas (seção 4.1)
- [x] Curva pulso × velocidade em malha aberta (mm/s): seção 5, figura `figuras/curva_aberta.png`
- [x] Comparação de dois filtros no mesmo degrau: seção 6 (escolhida a média móvel n = 5)
- [x] Sintonia do PID (tabela de tentativas): seção 7, ganhos finais kp = 3, ki = 8, kd = 0 (T06 com ki = 5 não foi feita)
- [x] Ensaios (seção 8): cinco patamares (100 a 160 mm/s), inversão e distância (+1,4 % sobre 3 voltas)

## Figuras (geradas com `uv run python scripts/graficos_ensaios.py`)

- `figuras/curva_aberta.png`: pulso × velocidade, com neutro, banda morta e limites de ±250 µs
- `figuras/filtros_degrau.png`: bruta, média móvel n = 5 e passa-baixa α = 0,3 (`scripts/grafico_filtros.py`)
- `figuras/pid_sintonia.png`: degraus T01 a T05 com setpoint de 150 mm/s
- `figuras/ensaio1_patamares.png` e `figuras/ensaio1_saturado.png`: cinco patamares, com o pulso no painel de baixo
- `figuras/ensaio2_inversao.png`: inversão de sentido, com o pulso
- `figuras/ensaio3_distancia.png`: s(t) com o valor esperado de três voltas e a velocidade

## Confirmações opcionais

- [ ] Se a roda ou a engrenagem do encoder foram mexidas ao trocar o motor, girar
  uma volta à mão (`z` antes, para zerar) e conferir se dá ≈ 287 ticks. Se der
  perto de 285 a 290, o N = 287,3 continua valendo; se desviar muito, repetir a
  medida de N.
- [ ] Anotar a data e as condições de cada ensaio (em especial a alimentação do
  servo: USB, fonte externa ou pilhas), porque o relatório declara a condição de
  cada ensaio (neutro, k, Δt, ganhos e parâmetro do filtro).
