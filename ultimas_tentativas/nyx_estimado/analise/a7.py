import numpy as np, sys; sys.path.insert(0, sys.argv[1]); from carregar import *
A = [carregar(m, l) for m, l, _ in partidas()]
rows = []
for d in A:
    P, H, C, F, t, j = d["P"], d["H"], d["C"], d["F"], d["t"], d["jogo"]
    for k in range(1, len(t)-1):
        if not j[k] or H[k, 0] or C[k, 0] or P[k, 0, 0] > 0: continue
        dt = t[k+1]-t[k]
        if not 0.04 < dt < 0.3: continue
        mv = np.hypot(*(P[k+1, 0, :2]-P[k, 0, :2])) / dt
        rows.append((mv < 0.02, C[k, 2] or C[k, 3], H[k, 2], np.hypot(*(F[k, 0]+[0.62, 0])) > 0.03, P[k, 0, 0], P[k, 0, 1],
                     np.hypot(*(P[k, 0, :2]-P[k, 2, :2])), P[k, 2, 0], t[k]-t[j][0]))
R = np.array(rows, float)
st = R[:, 0] == 1
print(f"atacante Nyx no próprio campo, sem bandeira: n={len(R)} parado {st.mean():.0%}")
for nome, m in [("bandeira do Nyx com o inimigo", R[:, 1] == 1), ("bandeira em casa", R[:, 1] == 0)]:
    print(f"  {nome:30s} n={m.sum():5d} parado={st[m].mean():.0%}")
s = R[st]
print("  parado: x,y p10/50/90", np.percentile(s[:, 4], [10, 50, 90]).round(2), np.percentile(np.abs(s[:, 5]), [10, 50, 90]).round(2),
      " dist ao atac. inimigo p50", np.median(s[:, 6]).round(2), " x do atac. inimigo p50", np.median(s[:, 7]).round(2))
print("  tempo de jogo quando parado p10/50/90", np.percentile(s[:, 8], [10, 50, 90]).round(1))
