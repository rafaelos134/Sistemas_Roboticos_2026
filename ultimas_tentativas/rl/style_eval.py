import sys, json
sys.path.insert(0, "/home/ubuntu/ros2_ws/rl")
from ctf_env import CtfEnv
from train_ctf import load_pool
model = sys.argv[1]; role = sys.argv[2] if len(sys.argv) > 2 else "att"
me = load_pool([model])
env = None
for st in range(5 if role == "att" else 4):
    w = [0.0] * (5 if role == "att" else 4); w[st] = 1.0
    kw = dict(def_styles=w) if role == "att" else dict(att_styles=w)
    if env is None:
        env = CtfEnv(n_envs=512, mode="attack" if role == "att" else "defend", episode_s=90.0, random_start=0.0,
                     easy_start=0.0, obs_noise=0.0, main_mix={"script": 1.0}, other_mix={"script": 1.0},
                     att_pool=load_pool([]), def_pool=load_pool([]), strong=1.0, **kw)
    else:
        env.att_styles = kw.get("att_styles", env.att_styles); env.def_styles = kw.get("def_styles", env.def_styles)
    obs = env.reset_all(); env.steps[:] = 0; env.ep_steps[:] = 0
    tot = {k: 0 for k in ("scored", "conceded", "pickup", "tagged")}
    for _ in range(env.max_steps):
        obs, rew, term, trunc, info = env.step(me.act(0, obs))
        for k in tot: tot[k] += int(info[k].sum())
    print(role, "estilo", st, {k: round(v / env.n, 2) for k, v in tot.items()}, flush=True)
