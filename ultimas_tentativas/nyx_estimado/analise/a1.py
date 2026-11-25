import numpy as np, sys; sys.path.insert(0, sys.argv[1]); from carregar import *
ms = partidas(); print(len(ms), "partidas do Nyx")
A = [carregar(m, l) for m, l, _ in ms]
# DEFENSOR (índice 1): posição em relação à base, quando a bandeira do Nyx está em casa e ninguém preso
X = []; 
for d in A:
    ok = d["jogo"] & ~d["H"][:, 1] & ~d["C"][:, 2] & ~d["C"][:, 3]
    for k in np.nonzero(ok)[0]:
        p = d["P"][k, 1]; oa = d["P"][k, 2]
        X.append([p[0], p[1], p[2], oa[0], oa[1], d["H"][k, 2]])
X = np.array(X); base = np.array([-0.62, 0])
r = np.hypot(X[:, 0] - base[0], X[:, 1] - base[1]); ang = np.arctan2(X[:, 1], X[:, 0] - base[0])
ra = np.hypot(X[:, 3] - base[0], X[:, 4] - base[1]); anga = np.arctan2(X[:, 4], X[:, 3] - base[0])
print("DEFENSOR (bandeira em casa): n", len(X))
print("  dist à base: p10/50/90", np.percentile(r, [10, 50, 90]).round(3), " x p50", np.median(X[:, 0]).round(3))
for lo, hi in [(0, .2), (.2, .35), (.35, .5), (.5, .8), (.8, 2)]:
    m = (ra >= lo) & (ra < hi) & (X[:, 5] == 0)
    if m.sum() > 50:
        dang = np.degrees(np.abs(np.angle(np.exp(1j*(ang[m]-anga[m])))))
        print(f"  atacante inimigo a {lo:.2f}-{hi:.2f} m da base: n={m.sum():5d} def_r p50={np.median(r[m]):.3f} "
              f"|ângulo def - ângulo atac| p50={np.median(dang):5.1f}°  dist def→atac p50={np.median(np.hypot(X[m,0]-X[m,3],X[m,1]-X[m,4])):.3f}")
# velocidade do defensor por situação
