"""Check that the exported soccer node computes the same observation and action as training
(torch) for random states, for both teams. Prints the largest differences."""
import sys

import torch

from eval_sumo import PARAMS, load_node
from soccer_env import SoccerEnv
from train_soccer import make_model

node_file, ckpt = sys.argv[1], sys.argv[2]
env = SoccerEnv(n_envs=64, random_start=1.0, easy_start=0.0, obs_noise=0.0)
for _ in range(10):
    env.step(torch.rand(128, 2, device="cuda") * 2 - 1)
rob, ball = env.state
model = make_model().cuda().eval()
model.load_state_dict(torch.load(ckpt, map_location="cuda", weights_only=False)["model"])
blue, red = env._team_views(rob, ball)
for team, view in (("blue", blue), ("red", red)):
    m = load_node(node_file, "node_" + team)
    PARAMS.clear(); PARAMS.update(team=team)
    node = m.Soccer()
    worst_o = worst_a = 0.0
    for e in range(env.n):
        node.poses = {i: tuple(rob[i, e, :3].tolist()) for i in range(6)}
        node.vel = {i: tuple(rob[i, e, 3:5].tolist()) for i in range(6)}
        node.ball, node.ball_v = tuple(ball[e, :2].tolist()), tuple(ball[e, 2:4].tolist())
        for role in (1, 2):
            node.prev = [(0.0, 0.0)] * 3
            mine = [node.robot(i) for i in node.ids]
            theirs = [node.robot(i) for i in node.opp_ids]
            s = -1.0 if node.flip else 1.0
            b = (s * node.ball[0], s * node.ball[1], s * node.ball_v[0], s * node.ball_v[1])
            o_np = torch.tensor(m.observation_np(mine, theirs, b, role, (0.0, 0.0)))
            o_t = env._obs_of(tuple(x[:, e:e + 1] if x.dim() == 3 else x[e:e + 1] for x in view), role,
                              torch.zeros(1, 2, device="cuda"))[0].cpu()
            worst_o = max(worst_o, (o_np - o_t).abs().max().item())
            a_np = torch.tensor(node.att(o_np.tolist()))
            with torch.no_grad():
                a_t = model.actor(model.norm(o_t.cuda().unsqueeze(0))).clamp(-1, 1)[0].cpu()
            worst_a = max(worst_a, (a_np - a_t).abs().max().item())
    print("%s: max |obs numpy - torch| = %.2e, max |action numpy - torch| = %.2e" % (team, worst_o, worst_a))
