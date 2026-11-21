# TP4 — Encoder e controle PID de velocidade

Sistemas Robóticos, UFMG.
Rafael Santos Oliveira, Vinicius Trindade Dias Abel, Guilherme Buxbaum Marinho Guerra, Thiago Henrique Silva de Almeida.

## Pastas

- `relatorio_final/`: relatório em PDF (`relatorio.pdf`) e anotações dos resultados (`anotacoes.md`).
- `dados_e_resultados/`: CSVs coletados nos ensaios e a curva em malha aberta (`curva_aberta.csv`).
- `scripts_controle/`: firmware do controle (`controle_pid_firmware.cpp`), teste de sanidade do servo e encoder (`teste_sanidade.cpp`) e scripts de análise dos CSVs.

## Dados

| Seção | Ensaio | CSV |
|---|---|---|
| 5 | Curva pulso × velocidade (malha aberta) | `malha_aberta_A50_3.csv`, `curva_aberta.csv` |
| 6 | Comparação de filtros | `filtros_degrau_o1700.csv` |
| 7 | Sintonia do PID | `pid_t01.csv` a `pid_t05.csv` |
| 8.1 | Cinco patamares | `ensaio1_patamares.csv`, `ensaio1_patamares_saturado.csv` |
| 8.2 | Inversão de sentido | `ensaio2_inversao.csv` |
| 8.3 | Distância | `ensaio3_distancia_v2.csv` (corrida final); `ensaio3_distancia.csv` e `_v1_com_malha` são a primeira corrida |

Velocidades em mm/s, pulso em µs, tempo em s.

## Scripts

- `curva_aberta.py`: gera `curva_aberta.csv` a partir de `malha_aberta_A50_3.csv`.
- `filtros.py`: compara os filtros no degrau (imprime as métricas).

```bash
python3 scripts_controle/curva_aberta.py
python3 scripts_controle/filtros.py
```
