import numpy as np, sys; sys.path.insert(0, sys.argv[1]); from carregar import *
ms = {m: l for m, l, _ in partidas()}
d = carregar(650, ms[650]); t = d["t"] - d["t"][d["jogo"]][0]; P = d["P"]
print(" t     x      y     yaw   v(m/s) w(rad/s) d_def  d_band  ang_band_rel")
for k in np.nonzero((t >= 4.0) & (t <= 9.5))[0]:
    me, od = P[k, 0], P[k, 3]; dt = d["t"][k+1]-d["t"][k]
    dx, dy = P[k+1, 0, :2]-me[:2]
    v = (dx*np.cos(me[2])+dy*np.sin(me[2]))/dt; w = wrap(P[k+1, 0, 2]-me[2])/dt
    fl = d["F"][k, 1]; ab = np.degrees(wrap(np.arctan2(fl[1]-me[1], fl[0]-me[0]) - me[2]))
    print(f"{t[k]:5.2f} {me[0]:+.3f} {me[1]:+.3f} {np.degrees(me[2]):+5.0f} {v:+.3f} {w:+6.2f} {np.hypot(*(me[:2]-od[:2])):.3f} {np.hypot(*(me[:2]-fl)):.3f} {ab:+5.0f}")
