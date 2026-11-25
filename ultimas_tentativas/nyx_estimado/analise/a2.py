import numpy as np, sys; sys.path.insert(0, sys.argv[1]); from carregar import *
A = [carregar(m, l) for m, l, _ in partidas()]
base = np.array([-0.62, 0.0])
saidas = []
tot = dict(n=0, fora=0)
for d in A:
    P, H, C, t, j = d["P"], d["H"], d["C"], d["t"], d["jogo"]
    r = np.hypot(P[:, 1, 0] - base[0], P[:, 1, 1])
    fora = (r > 0.05) & j
    tot["n"] += j.sum(); tot["fora"] += fora.sum()
    k = 1
    while k < len(t):
        if fora[k] and not fora[k-1] and j[k-1]:
            k0 = k; 
            while k < len(t) and fora[k]: k += 1
            seg = slice(k0, k)
            oa = P[k0-1, 2]; od = P[k0-1, 3]
            saidas.append(dict(mid=d["mid"], t=t[k0]-t[j][0], dur=t[min(k, len(t)-1)]-t[k0],
                rmax=r[seg].max(), xmax=P[seg, 1, 0].max(),
                opp_att_dist=np.hypot(*(oa[:2]-P[k0-1, 1, :2])), opp_att_x=oa[0], opp_att_held=H[k0-1, 2],
                opp_carry=C[k0-1, 2] or C[k0-1, 3], my_att_held=H[k0-1, 0], my_att_carry=C[k0-1, 0],
                flag_home=np.hypot(*(d["F"][k0-1, 0]-base)) < 0.03,
                end_tag=H[min(k, len(t)-1), 2] or H[min(k, len(t)-1), 3]))
        k += 1
print(f"defensor fora da base (>5 cm): {tot['fora']/tot['n']:.1%} do tempo de jogo; {len(saidas)} saídas")
import collections
def resumo(nome, sel):
    s = [x for x in saidas if sel(x)]
    if not s: return
    print(f"  {nome:42s} n={len(s):3d} dur p50={np.median([x['dur'] for x in s]):5.1f}s rmax p50={np.median([x['rmax'] for x in s]):.2f} "
          f"dist atac.inimigo na saída p50={np.median([x['opp_att_dist'] for x in s]):.2f} termina segurando={np.mean([x['end_tag'] for x in s]):.0%}")
resumo("todas", lambda x: True)
resumo("bandeira do Nyx carregada pelo inimigo", lambda x: x["opp_carry"])
resumo("bandeira em casa, atac. inimigo solto", lambda x: not x["opp_carry"] and not x["opp_att_held"])
resumo("bandeira em casa, atac. inimigo PRESO", lambda x: not x["opp_carry"] and x["opp_att_held"])
resumo("nosso atac. (Nyx) carregando", lambda x: x["my_att_carry"])
resumo("nosso atac. (Nyx) preso", lambda x: x["my_att_held"])
print("saídas longas (>3 s):")
for x in sorted(saidas, key=lambda x: -x["dur"])[:8]:
    print("   ", {k: (round(float(v), 2) if not isinstance(v, (bool, np.bool_)) else bool(v)) for k, v in x.items()})
