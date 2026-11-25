import numpy as np, sys; sys.path.insert(0, sys.argv[1]); from carregar import *
A = [carregar(m, l) for m, l, _ in partidas()]
orb, wait, ysame = [], [], []
for d in A:
    P, H, C, t, j = d["P"], d["H"], d["C"], d["t"], d["jogo"]
    near = j & (P[:, 0, 0] > 0.45) & ~C[:, 0]
    w = j & (np.abs(P[:, 0, 0] + 0.10) < 0.06) & (np.abs(np.abs(P[:, 0, 1]) - 0.33) < 0.06)
    # sequência: perto da bandeira (contínuo) -> espera ; mede quanto tempo ficou perto antes de ir esperar
    k = 1
    while k < len(t):
        if near[k] and not near[k-1]:
            k0 = k
            while k < len(t) and near[k]: k += 1
            k1 = min(k, len(t)-1)
            # depois de sair, foi para o ponto de espera em < 6 s?
            k2 = np.searchsorted(t, t[k1] + 6.0)
            went = w[k1:k2].any()
            held = H[k0:k1+1, 0].any()
            orb.append((t[k1]-t[k0], went, held, C[k1, 0]))
            if went:
                kw = k1 + np.argmax(w[k1:k2]); ysame.append(np.sign(P[kw, 0, 1]) == np.sign(P[k1-1, 0, 1]))
        k += 1
o = np.array(orb, float)
for nome, m in [("saiu e foi esperar", o[:, 1] == 1), ("saiu sem ir esperar", o[:, 1] == 0)]:
    print(f"{nome:24s} n={m.sum():3.0f}  tempo perto da bandeira p10/50/90 {np.percentile(o[m, 0], [10, 50, 90]).round(1)}  "
          f"foi preso {o[m, 2].mean():.0%}  saiu com a bandeira {o[m, 3].mean():.0%}")
print("lado (y) da espera = lado de onde estava:", np.mean(ysame).round(2) if ysame else None)
