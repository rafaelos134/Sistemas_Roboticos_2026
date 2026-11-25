import numpy as np, sys; sys.path.insert(0, sys.argv[1]); from carregar import *
A = [carregar(m, l) for m, l, _ in partidas()]
segs = []
for d in A:
    P, H, C, F, t, j = d["P"], d["H"], d["C"], d["F"], d["t"], d["jogo"]
    w = j & (np.abs(P[:, 0, 0] + 0.10) < 0.06) & (np.abs(np.abs(P[:, 0, 1]) - 0.33) < 0.06)
    k = 1
    while k < len(t):
        if w[k] and not w[k-1]:
            k0 = k
            while k < len(t) and w[k]: k += 1
            k1 = min(k, len(t)-1)
            # o que veio antes (2 s antes de chegar): onde estava
            kb = max(0, np.searchsorted(t, t[k0]-3.0))
            segs.append(dict(dur=t[k1]-t[k0], antes_x=P[kb, 0, 0], antes_preso=H[kb:k0, 0].any(),
                             reset=np.abs(np.diff(P[kb:k0+1, 0, 0])).max() > 0.08 if k0 > kb else False,
                             saida_inim_att_x=P[k1, 2, 0], saida_inim_att_preso=H[k1, 2], saida_inim_def_x=P[k1, 3, 0],
                             saida_inim_def_preso=H[k1, 3], chegada_inim_att_x=P[k0, 2, 0], chegada_inim_att_preso=H[k0, 2],
                             saida_band_nyx_fora=np.hypot(*(F[k1, 0]+[0.62, 0])) > 0.03, saida_inim_att_carrega=C[k1, 2]))
        k += 1
import pandas as pd
df = pd.DataFrame(segs); df = df[df.dur > 0.5]
print(len(df), "esperas >0,5 s; duração p50", df.dur.median().round(1), "s")
print(df.mean(numeric_only=True).round(2).to_string())
print("\nna CHEGADA: atac. inimigo preso?", df.chegada_inim_att_preso.mean().round(2), " x do atac. inimigo p50", df.chegada_inim_att_x.median().round(2))
print("na SAÍDA:   atac. inimigo preso?", df.saida_inim_att_preso.mean().round(2), " x p50", df.saida_inim_att_x.median().round(2))
