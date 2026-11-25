"""Parity check: the numpy GRU policy of the exported node == ppo_rnn.RecurrentActorCritic (torch)."""
import subprocess
import sys
from pathlib import Path

import torch

import eval_ctf  # noqa: F401  (stubs rospy)
from ctf_obs import OBS_DIM
from eval_sumo import load_node
from ppo_rnn import RecurrentActorCritic
from train_ctf import HIDDEN

HERE = Path(__file__).resolve().parent
m = RecurrentActorCritic(OBS_DIM, 2, hidden=HIDDEN)
m.load_state_dict(torch.load(HERE / "runs/ctf_att_4/model.pt", weights_only=False)["model"], strict=False)
torch.nn.init.normal_(m.head.weight, std=0.3)
tmp = HERE / "runs" / "_parity"
tmp.mkdir(exist_ok=True)
torch.save({"model": m.state_dict()}, tmp / "rnn.pt")
subprocess.run([sys.executable, str(HERE / "export_ctf.py"), "--att", str(tmp / "rnn.pt"), "--def-script",
                "--out", str(tmp / "node.py")], check=True)
mod = load_node(str(tmp / "node.py"), "parity_node")
pol = mod.Policy("att_", 3)
h = m.init_hidden(1, "cpu")
err = 0.0
torch.manual_seed(0)
for t in range(200):
    o = torch.randn(1, OBS_DIM) * 2
    with torch.no_grad():
        a, h = m.mean_h(m.norm(o), h)
    b = pol(o[0].numpy())
    err = max(err, float(abs(a[0].numpy() - b).max()))
print("max |torch - numpy| over 200 recurrent steps: %.2e" % err)
