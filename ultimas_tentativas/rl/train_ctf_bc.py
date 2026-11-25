"""Behavior cloning of the REAL tournament teams (dataset from live_ctf.py): one network per team and role,
in the same format as the PPO networks (train_ctf.make_model, checkpoint {"model": state_dict}), so they go
straight into the opponent pools of train_ctf.py / train_ctf_team.py / eval_ctf_vec.py.

    $PY train_ctf_bc.py                         # all teams except ours (R2D2) and unknown ("?")
    $PY train_ctf_bc.py --teams "Grupo 6 · SoLaDaBot"
    $PY train_ctf_bc.py --all                   # plus one network per role with all opponents together

Writes runs/bc_<team>_<att|def>/model.pt and runs/bc_summary.json. Validation: whole matches held out
(15 %), so the number says how well it imitates a team in a match it has not seen.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import torch

from train_ctf import make_model


def slug(name):
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = s.split("·")[-1] if "·" in name else s
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return s or "x"


def train_one(obs, act, match, epochs, seed, label):
    g = np.random.default_rng(seed)
    ms = np.unique(match)
    val_m = set(g.choice(ms, size=max(1, int(round(0.15 * len(ms)))), replace=False).tolist()) if len(ms) > 2 else set()
    val = np.isin(match, list(val_m))
    tr = ~val
    dev = "cuda"
    X, Y = torch.tensor(obs, device=dev), torch.tensor(act, device=dev)
    model = make_model().to(dev)
    with torch.no_grad():  # normalization from the training data (as RunningNorm would have it)
        model.norm.mean.copy_(X[tr].mean(0))
        model.norm.var.copy_(X[tr].var(0) + 1e-6)
        model.norm.count.fill_(float(tr.sum()))
        model.log_std.fill_(float(np.log(0.3)))
    opt = torch.optim.Adam(model.actor.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    itr = torch.tensor(np.nonzero(tr)[0], device=dev)
    iva = torch.tensor(np.nonzero(val)[0], device=dev)

    def loss_on(idx):
        pred = model.actor(model.norm(X[idx])).clamp(-1.5, 1.5)
        return torch.nn.functional.smooth_l1_loss(pred, Y[idx], beta=0.2)

    def metrics(idx):
        if idx.numel() == 0:
            return float("nan"), float("nan")
        with torch.no_grad():
            pred = model.actor(model.norm(X[idx])).clamp(-1, 1)
            mae = (pred - Y[idx]).abs().mean().item()
            sign = ((pred[:, 0] > 0) == (Y[idx][:, 0] > 0)).float().mean().item()
        return mae, sign

    base = (Y[iva] - Y[itr].mean(0)).abs().mean().item() if iva.numel() else float("nan")
    for ep in range(epochs):
        perm = itr[torch.randperm(itr.numel(), device=dev)]
        for b in perm.split(4096):
            opt.zero_grad()
            loss_on(b).backward()
            opt.step()
        sched.step()
    mae_tr, _ = metrics(itr)
    mae_va, sign_va = metrics(iva)
    print(f"  {label:34s} n={int(tr.sum()):6d}/{int(val.sum()):5d}  MAE treino {mae_tr:.3f}  validação {mae_va:.3f} "
          f"(média constante {base:.3f})  sinal de v certo {sign_va:.0%}", flush=True)
    return model, {"n_train": int(tr.sum()), "n_val": int(val.sum()), "mae_train": mae_tr, "mae_val": mae_va,
                   "mae_const": base, "sign_val": sign_va, "val_matches": sorted(int(m) for m in val_m)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/live_ctf.npz")
    p.add_argument("--teams", nargs="*", default=None)
    p.add_argument("--skip", nargs="*", default=["Grupo 4 · R2D2", "?"])
    p.add_argument("--all", action="store_true", help="also one network per role with every opponent team")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    d = np.load(a.data)
    names = list(d["team_names"])
    obs, act, team, role, match = d["obs"], d["act"], d["team"], d["role"], d["match"]
    teams = a.teams or [n for n in names if n not in a.skip]
    groups = [(n, team == names.index(n)) for n in teams]
    if a.all:
        groups.append(("todos", np.isin(team, [names.index(n) for n in teams])))
    summary = {}
    for name, sel in groups:
        for r, rn in ((0, "att"), (1, "def")):
            m = sel & (role == r)
            if m.sum() < 2000:
                print(f"  {name} {rn}: poucas amostras ({int(m.sum())}), pulando")
                continue
            model, info = train_one(obs[m], act[m], match[m], a.epochs, a.seed, f"{name} {rn}")
            out = Path("runs") / f"bc_{slug(name)}_{rn}"
            out.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "role": rn, "team": name, "bc": info}, out / "model.pt")
            summary[f"{name}|{rn}"] = {"path": str(out / "model.pt"), **info}
    with open("runs/bc_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("salvo: runs/bc_*/model.pt e runs/bc_summary.json")


if __name__ == "__main__":
    main()
