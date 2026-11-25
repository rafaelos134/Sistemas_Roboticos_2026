import numpy as np, sys; sys.path.insert(0, sys.argv[1]); from carregar import *
A = [carregar(m, l) for m, l, _ in partidas()]
dmin_free, near_flag_free, held_rush, pick_dt, home_paths = [], [], [], [], []
for d in A:
    P, H, C, F, t, j = d["P"], d["H"], d["C"], d["F"], d["t"], d["jogo"]
    att, od = P[:, 0], P[:, 3]
    dd = np.hypot(att[:, 0]-od[:, 0], att[:, 1]-od[:, 1]); df = np.hypot(att[:, 0]-F[:, 1, 0], att[:, 1]-F[:, 1, 1])
    free = j & ~H[:, 0] & ~H[:, 3] & ~C[:, 0] & (od[:, 0] > 0) & (att[:, 0] > 0.3)
    dmin_free += dd[free].tolist(); near_flag_free += df[free].tolist()
    # quando o defensor inimigo fica preso, quanto tempo até o atacante do Nyx pegar a bandeira?
    for k in range(1, len(t)):
        if j[k] and H[k, 3] and not H[k-1, 3]:
            k2 = k
            while k2 < len(t) and H[k2, 3] and not C[k2, 0]: k2 += 1
            pick_dt.append((t[k2]-t[k]) if k2 < len(t) and C[k2, 0] else np.nan)
    # trajeto com a bandeira: reto para casa?
    for k in range(1, len(t)):
        if C[k, 0] and not C[k-1, 0]:
            k2 = k
            while k2 < len(t)-1 and C[k2, 0]: k2 += 1
            path = np.hypot(*np.diff(att[k:k2+1, :2], axis=0).T).sum(); straight = np.hypot(*(att[k2, :2]-att[k, :2]))
            home_paths.append((straight / max(path, 1e-6), t[k2]-t[k], H[k2, 0], d["score"][k2][0] > d["score"][k][0] if k2 + 1 < len(t) else False))
dmin_free = np.array(dmin_free); near = np.array(near_flag_free)
print("atacante Nyx perto da bandeira inimiga, defensor inimigo solto:")
print("  dist ao defensor p1/p5/p25/p50:", np.percentile(dmin_free, [1, 5, 25, 50]).round(3), "  (segurar = 0.10)")
print("  dist à bandeira p5/p50:", np.percentile(near, [5, 50]).round(3), "  (pegar = 0.07)")
pk = np.array(pick_dt); print(f"defensor inimigo preso {len(pk)} vezes; Nyx pegou a bandeira em {np.mean(~np.isnan(pk)):.0%} delas, mediana {np.nanmedian(pk):.1f} s depois")
hp = np.array(home_paths, float)
print(f"com a bandeira: {len(hp)} vezes; reta/caminho p50={np.median(hp[:,0]):.2f}; termina preso {hp[:,2].mean():.0%}; marca {hp[:,3].mean():.0%}")
