"""Convert the log.csv of a finished run (trained before TensorBoard was added) into
TensorBoard events, so every run shows up in the same dashboard.

    ~/ros2_ws/.venv/bin/python csv_to_tb.py runs/sumo_ppo_v1 runs/sumo_ppo_noshaping
"""
import csv
import sys
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

TAGS = {"win": "resultado/vitoria", "lose": "resultado/derrota", "draw": "resultado/empate",
        "ep_len_s": "episodio/duracao_s", "reward_step": "episodio/recompensa_por_passo",
        "pi_loss": "ppo/perda_actor", "v_loss": "ppo/perda_critic", "entropy": "ppo/entropia", "kl": "ppo/kl",
        "lr": "ppo/lr", "std_lin": "ppo/std_linear", "std_ang": "ppo/std_angular", "fps": "treino/passos_por_s"}

for run in sys.argv[1:]:
    run = Path(run)
    if (run / "tb").exists():
        print("skip %s: it already has tb/" % run)
        continue
    tb = SummaryWriter(str(run / "tb"))
    for row in csv.DictReader(open(run / "log.csv")):
        step = int(row["steps"])
        for col, tag in TAGS.items():
            if col in row:
                tb.add_scalar(tag, float(row[col]), step)
        for col, val in row.items():
            if col.startswith("win_") and int(row.get("n_" + col[4:], 0)) > 0:
                tb.add_scalar("vitoria_por_adversario/" + col[4:], float(val), step)
    tb.close()
    print("converted", run)
