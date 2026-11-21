import csv
import math
import os
import statistics as st

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
D_MM, N_TICKS = 43.40, 287.3
K = math.pi * D_MM / N_TICKS
NEUTRO_US = 1495
DESCARTE_S = 1.5

grupos = {}
with open(os.path.join(RAIZ, "dados_e_resultados", "malha_aberta_A50_3.csv")) as f:
    for r in csv.DictReader(f):
        grupos.setdefault(int(float(r["pulso_us"])), []).append(
            (float(r["t_s"]), float(r["ticks"]), float(r["v_bruta"])))

saida = []
for pulso, L in grupos.items():
    t0 = L[0][0]
    reg = [x for x in L if x[0] >= t0 + DESCARTE_S]
    dtk = reg[-1][1] - reg[0][1]
    dt = reg[-1][0] - reg[0][0]
    saida.append((pulso, K * dtk / dt, st.stdev(x[2] for x in reg)))

with open(os.path.join(RAIZ, "dados_e_resultados", "curva_aberta.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["pulso_us", "v_regime_mm_s", "desvio_v_bruta_mm_s"])
    for p, v, s in saida:
        w.writerow([p, f"{v:.1f}", f"{s:.1f}"])
        print(f"{p:5d}  {v:8.1f}  {s:5.1f}")
