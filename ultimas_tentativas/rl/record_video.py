"""Record a sumo match of two kit nodes in psr_sim as an MP4 (offscreen camera, no window).

    PY=~/ros2_ws/.venv/bin/python; S=exports/sumo
    $PY record_video.py $S/sumo_rl_v3.py --opponent $S/sumo_flanker.py --rounds 3 --title "Fase 3 x flanker" \
        --out docs/videos/fase3_vs_flanker.mp4

Same driving code as eval_sumo.py (the node files are imported and run as in ROS).
A Genesis camera renders the scene at 30 frames per simulated second (the video plays
in real time); the scoreboard, the referee message and the round clock are drawn on top.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import eval_sumo
from eval_sumo import NodeDriver

FPS = 30
W, H = 1280, 720


def font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("node")
    p.add_argument("--opponent", default=None)
    p.add_argument("--side", type=int, default=0, choices=(0, 1))
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--max-time", type=float, default=135.0, help="stop after this many simulated seconds")
    p.add_argument("--title", default="")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import genesis as gs
    from psr_sim.scenarios import SCENARIOS
    from psr_sim.world import World

    cams = []
    build = gs.Scene.build

    def build_with_camera(scene, *a, **k):  # add a camera before the scene is built
        cams.append(scene.add_camera(res=(W, H), pos=(0.0, -0.95, 0.85), lookat=(0.0, 0.02, 0.0), fov=40, GUI=False))
        return build(scene, *a, **k)

    gs.Scene.build = build_with_camera

    a_i, b_i = args.side, 1 - args.side
    drivers = [(a_i, NodeDriver(args.node, a_i, b_i, "node_a"))]
    if args.opponent:
        drivers.append((b_i, NodeDriver(args.opponent, b_i, a_i, "node_b")))
    names = {a_i: Path(args.node).stem, b_i: Path(args.opponent).stem if args.opponent else "parado"}
    sc = SCENARIOS["sumo"](2, args.seed)
    w = World(sc, backend="gpu")
    cam = cams[0]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-crf", "23", "-preset", "veryfast", "-movflags", "+faststart", str(out)],
                          stdin=subprocess.PIPE)
    f_big, f_mid, f_small = font(34), font(26), font(20)

    next_t = {i: 0.0 for i, _ in drivers}
    next_frame, round_t0 = 0.0, w.t
    rounds = 0
    last = (0, 0, 0)
    state = w.read()
    while rounds < args.rounds and w.t < args.max_time:
        if w.pending is None:
            for i, drv in drivers:
                if w.t >= next_t[i]:
                    w.set_cmd(i, *drv.act(state["robots"], w.t))
                    next_t[i] = w.t + 1.0 / drv.rate
        was_pending = w.pending is not None
        state = w.tick()
        if was_pending and w.pending is None:
            for _, drv in drivers:
                drv.new_round()
            round_t0 = w.t
        score = (sc.wins[0], sc.wins[1], sc.draws)
        if score != last:
            rounds += 1
            last = score
        if w.t >= next_frame:
            next_frame += 1.0 / FPS
            rgb = cam.render()[0]
            img = Image.fromarray(np.ascontiguousarray(rgb[..., :3]).astype(np.uint8))
            d = ImageDraw.Draw(img)
            d.rectangle([0, 0, W, 92], fill=(255, 255, 255))
            d.text((20, 10), args.title, font=f_big, fill=(20, 20, 30))
            board = "%s (azul) %d  x  %d %s (laranja)    empates: %d" % (
                names[0], sc.wins[0], sc.wins[1], names[1], sc.draws)
            d.text((20, 54), board, font=f_small, fill=(40, 40, 60))
            rt = "rodada %d   %4.1f s" % (min(rounds + (w.pending is None), args.rounds), w.t - round_t0)
            d.text((W - 260, 54), rt, font=f_small, fill=(40, 40, 60))
            if w.message:
                tw = d.textlength(w.message, font=f_mid)
                d.rectangle([(W - tw) / 2 - 16, H - 80, (W + tw) / 2 + 16, H - 30], fill=(150, 10, 33))
                d.text(((W - tw) / 2, H - 76), w.message, font=f_mid, fill=(255, 255, 255))
            ff.stdin.write(img.tobytes())
    # hold the last frame for 1 s
    for _ in range(FPS):
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()
    w.close()
    print("saved %s: %s %d x %d %s, draws %d (%.0f s)" % (out, names[0], sc.wins[0], sc.wins[1], names[1], sc.draws, w.t))


if __name__ == "__main__":
    main()
