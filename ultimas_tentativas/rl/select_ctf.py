"""Pick a checkpoint by real matches in psr_sim (not by the training statistics).

Every checkpoint is exported as a kit node and plays short matches as blue AND as red against a fixed set
of opponent nodes; the score is the goal difference (ties broken by pickups allowed / made).

    $PY select_ctf.py --role att --cks runs/ctf_att_5/ck_*.pt runs/ctf_att_5/model.pt --partner script
    $PY select_ctf.py --role def --cks runs/ctf_def_5/ck_*.pt --partner runs/ctf_att_5/model.pt
    $PY select_ctf.py --role team --cks runs/ctf_team_1/att.pt:runs/ctf_team_1/def.pt runs/ctf_att_5/model.pt:script
--role team: every --cks item is "attacker:defender" (defender "script" = hybrid node), --partner is not used.
--partner: the other role of the node (a network, or "script" for the scripted defender = hybrid node).
Writes exports/ctf/sel/<run>_<ck>.py, results/select_<role>_<run>.csv and prints the ranking.
"""
import argparse
import csv
import subprocess
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPPS = ["parado", "exports/ctf/ctf_script.py", "exports/ctf/ctf_hibrido_v4.py", "exports/ctf/ctf_rl_v4.py",
        "../classico/ctf/baseline.py", "../classico/ctf/ctf_parado.py", "../classico/ctf/ctf.py"]


def export(role, ck, partner, out):
    if role == "team":
        att, dfn = ck.split(":")
        cmd = ["--att", att] + (["--def-script"] if dfn == "script" else ["--def", dfn])
    elif role == "att":
        cmd = ["--att", ck] + (["--def-script"] if partner == "script" else ["--def", partner])
    else:
        cmd = ["--att", partner, "--def", ck]
    subprocess.run([sys.executable, str(HERE / "export_ctf.py")] + cmd + ["--out", out], check=True,
                   capture_output=True)


def match(blue, red, minutes, seed):
    import eval_ctf
    a = types.SimpleNamespace(blue=None if blue == "parado" else blue, red=None if red == "parado" else red,
                              minutes=minutes, viewer=False)
    try:
        return eval_ctf._match_subprocess(a, seed)
    except subprocess.CalledProcessError as e:  # an opponent node that does not run
        print("  falhou: %s x %s (%s)" % (blue, red, (e.stderr or "").strip().splitlines()[-1:]), flush=True)
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", required=True, choices=("att", "def", "team"))
    p.add_argument("--cks", nargs="+", required=True)
    p.add_argument("--nodes", action="store_true", help="--cks are already exported node files (no export)")
    p.add_argument("--partner", default=None)
    p.add_argument("--opps", nargs="*", default=OPPS)
    p.add_argument("--minutes", type=float, default=3.0)
    p.add_argument("--seeds", type=int, default=1, help="matches per opponent and side")
    p.add_argument("--jobs", type=int, default=3)
    a = p.parse_args()
    nodes = []
    for ck in a.cks:
        if a.nodes:
            nodes.append(ck)
            continue
        if a.role == "team":
            at, df = ck.split(":")
            out = "exports/ctf/sel/%s_%s__%s.py" % (Path(at).parent.name, Path(at).stem,
                                                   "script" if df == "script" else Path(df).parent.name + "_" + Path(df).stem)
        else:
            out = "exports/ctf/sel/%s_%s.py" % (Path(ck).parent.name, Path(ck).stem)
        export(a.role, ck, a.partner, out)
        nodes.append(out)
    jobs = [(n, o, side, s) for n in nodes for o in a.opps for side in ("blue", "red") for s in range(a.seeds)]
    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(lambda j: match(j[0], j[1], a.minutes, 11 + j[3]) if j[2] == "blue"
                          else match(j[1], j[0], a.minutes, 11 + j[3]), jobs))
    rows = {n: {"node": n, "gf": 0, "ga": 0, "pf": 0, "pa": 0, "fails": 0} for n in nodes}
    per = {}
    for (n, o, side, s), r in zip(jobs, res):
        if r is None:
            rows[n]["fails"] += 1
            continue
        me, ot = ("blue", "red") if side == "blue" else ("red", "blue")
        gf, ga = r[me], r[ot]
        rows[n]["gf"] += gf
        rows[n]["ga"] += ga
        rows[n]["pf"] += r["pickups_" + me]
        rows[n]["pa"] += r["pickups_" + ot]
        per.setdefault((n, o), [0, 0])
        per[(n, o)][0] += gf
        per[(n, o)][1] += ga
    key = (lambda r: (r["gf"] - r["ga"], -r["pa"], r["pf"])) if a.role == "def" else \
          (lambda r: (r["gf"] - r["ga"], r["pf"], -r["pa"]))
    ranked = sorted(rows.values(), key=key, reverse=True)
    short = lambda o: Path(o).stem  # noqa: E731
    print("%-34s %6s %6s | " % ("checkpoint", "saldo", "pega") + " ".join("%14s" % short(o)[:14] for o in a.opps))
    for r in ranked:
        print("%-34s %+6d %3d/%-3d| " % (Path(r["node"]).stem, r["gf"] - r["ga"], r["pf"], r["pa"])
              + " ".join("%14s" % ("%d x %d" % tuple(per.get((r["node"], o), (0, 0)))) for o in a.opps)
              + ("  (%d falhas)" % r["fails"] if r["fails"] else ""))
    name = Path(a.cks[0].split(":")[0]).parent.name
    with open(HERE / "results" / ("select_%s_%s.csv" % (a.role, name)), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node", "goal_diff", "gf", "ga", "pickups_for", "pickups_against"] + [short(o) for o in a.opps])
        for r in ranked:
            w.writerow([r["node"], r["gf"] - r["ga"], r["gf"], r["ga"], r["pf"], r["pa"]]
                       + ["%d-%d" % tuple(per.get((r["node"], o), (0, 0))) for o in a.opps])
    print("MELHOR", ranked[0]["node"])


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
