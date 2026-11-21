# Resumo dos resultados — TP4

## Encoder

| Grandeza | Valor |
|---|---|
| Diâmetro da roda (paquímetro) | D = 43,40 mm |
| Ticks por volta (média de voltas à mão, biblioteca Encoder) | N = 287,3 ± 0,57 |
| Avanço por tick | k = π·D/N = 0,4746 ± 0,0011 mm/tick |
| Período de amostragem | Δt = 50 ms |
| Degrau de quantização da velocidade | k/Δt = 9,5 mm/s |
| Neutro do servo | 1495 µs |

## Malha aberta (`curva_aberta.csv`)

- Velocidade em regime de −258 mm/s (1000 µs) a +273 mm/s (2000 µs), quase linear.
- Banda morta em torno de 1500 µs: parado de 1470 a 1520 µs; em 1450 e 1550 µs já gira a −82 e +92 mm/s.
- Ruído da velocidade bruta: 5 a 16 mm/s.

## Filtros (`filtros_degrau_o1700.csv`)

Degrau de 0 a 152 mm/s em malha aberta.

| Filtro | Desvio em regime (mm/s) | t63 (s) | t90 (s) |
|---|---|---|---|
| Bruta | 13,2 | 0,08 | 0,12 |
| Média móvel n = 5 | 4,5 | 0,21 | 0,32 |
| Passa-baixa α = 0,3 | 4,2 | 0,21 | 0,44 |

Escolhida a média móvel com n = 5.

## Sintonia do PID (degrau de 0 a 150 mm/s)

| Teste | kp | ki | kd | v em regime (mm/s) | Observação |
|---|---|---|---|---|---|
| T01 | 1 | 0 | 0 | 87,6 | erro de regime de 42 % |
| T02 | 3 | 0 | 0 | 110,3 | erro de regime de 26 % |
| T03 | 6 | 0 | 0 | 123,9 | começa a oscilar (Ku ≈ 6, Tu ≈ 0,35 s) |
| T04 | 3 | 3 | 0 | 145,1 | ainda subindo em 4 s |
| T05 | 3 | 8 | 0 | 150,0 | erro zero, sobressinal de 7,6 % |

Ganhos finais: **kp = 3, ki = 8, kd = 0**. A correção do PID é limitada a ±250 µs (pulso entre 1245 e 1745 µs).

## Ensaios em malha fechada

**Ensaio 1, cinco patamares de 3 s** (`ensaio1_patamares.csv`)

| Patamar (mm/s) | 100 | 115 | 130 | 145 | 160 |
|---|---|---|---|---|---|
| v média (mm/s) | 100,0 | 114,4 | 129,2 | 144,7 | 160,2 |
| Erro | 0,0 % | 0,5 % | 0,6 % | 0,2 % | 0,1 % |

Sobressinal de 31 % no primeiro degrau (saturação do pulso) e de 3 a 6 % nos demais.

`ensaio1_patamares_saturado.csv`: com patamares de 190 e 220 mm/s o pulso satura em 1745 µs e a roda fica em ≈ 167 mm/s. A faixa útil da malha vai de ≈ 80 a ≈ 165 mm/s.

**Ensaio 2, inversão de sentido** (`ensaio2_inversao.csv`): de +120 para −120 mm/s, regime de 120,1 e −120,6 mm/s. A roda passa por zero 0,15 s depois da troca, com sobressinal de 31 %.

**Ensaio 3, distância de 3 voltas** (alvo 409,04 mm): 414,78 mm (874 ticks contra 862), erro de +1,4 % (`ensaio3_distancia_v2.csv`). O pulso é cortado para o neutro ao atingir o alvo. A primeira corrida (`ensaio3_distancia_v1_com_malha.csv`, igual a `ensaio3_distancia.csv`) manteve a malha ativa depois do alvo e a roda ficou oscilando em torno do zero, então não serve como medida de distância.
