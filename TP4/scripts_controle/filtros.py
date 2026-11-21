import csv
import os
import statistics as st

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS = 0.05
with open(os.path.join(RAIZ, "dados_e_resultados", "filtros_degrau_o1700.csv")) as f:
    d = [(float(r["t_s"]), float(r["v_bruta"])) for r in csv.DictReader(f)]
t = [x[0] for x in d]
raw = [x[1] for x in d]
reg = [i for i, x in enumerate(t) if x >= 5]
vfin = st.mean(raw[i] for i in reg)


def media_movel(n):
    buf, i, out = [0.0] * n, 0, []
    for v in raw:
        buf[i] = v
        i = (i + 1) % n
        out.append(sum(buf) / n)
    return out


def passa_baixa(a):
    y, out = 0.0, []
    for v in raw:
        y += a * (v - y)
        out.append(y)
    return out


def cruza(y, frac):
    lim = frac * vfin
    for i in range(1, len(y)):
        if y[i] >= lim > y[i - 1]:
            return t[i - 1] + TS * (lim - y[i - 1]) / (y[i] - y[i - 1])


def metricas(nome, y):
    sd = st.stdev(y[i] for i in reg)
    sdd = st.stdev((y[i] - y[i - 1]) / TS for i in reg)
    pico = max(y[i] for i, x in enumerate(t) if x < 5)
    print(f"{nome:10s} sd={sd:6.2f} mm/s  sd(dv/dt)={sdd:7.1f}  "
          f"t63={cruza(y, .632):.3f} s  t90={cruza(y, .9):.3f} s  pico={pico:.1f}")


print(f"v final = {vfin:.2f} mm/s")
metricas("bruta", raw)
for n in (3, 5, 8, 10):
    metricas(f"MM n={n}", media_movel(n))
for a in (0.15, 0.3, 0.5):
    metricas(f"LP a={a}", passa_baixa(a))
