"""Record a capture-the-flag match (2 x 2) of two kit nodes in psr_sim as MP4 (offscreen camera).

    $PY record_ctf.py --blue exports/ctf/ctf_rl.py --red exports/ctf/ctf_script.py \
        --seconds 90 --title "..." --out docs/videos/ctf/x.mp4
Without --red the red team stays stopped (red:=none). Same driver as eval_ctf.py.
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from eval_ctf import TEAMS, RobotDriver
from record_video import FPS, font

W, H = 1280, 720


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--blue", default=None)
    p.add_argument("--red", default=None)
    p.add_argument("--seconds", type=float, default=90)
    p.add_argument("--title", default="")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    import genesis as gs
    from psr_sim.scenarios import SCENARIOS
    from psr_sim.world import World

    cams, build = [], gs.Scene.build

    def build_cam(scene, *x, **k):
        cams.append(scene.add_camera(res=(W, H), pos=(0.0, -1.55, 1.75), lookat=(0.0, 0.05, 0.0), fov=45, GUI=False))
        return build(scene, *x, **k)

    gs.Scene.build = build_cam
    drivers = []
    for team, path in (("blue", a.blue), ("red", a.red)):
        if path:
            drivers += [RobotDriver(path, i, team, role) for i, role in TEAMS[team]]
    name = lambda x: Path(x).stem if x else "parado"  # noqa: E731
    names = [name(a.blue), name(a.red)]
    sc = SCENARIOS["ctf"](4, a.seed)
    w = World(sc, backend="gpu")
    cam = cams[0]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H),
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset",
                           "veryfast", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    fb, fs, fm = font(32), font(20), font(26)
    next_t = next_f = 0.0
    pickups, last_carrier = [0, 0], [None, None]
    state = w.read()
    while w.t < a.seconds:
        if w.pending is None and w.t >= next_t:
            for d in drivers:
                w.set_cmd(d.robot, *d.act(state["robots"], sc, w.t))
            next_t = w.t + 0.05
        state = w.tick()
        for team in (0, 1):
            c = sc.carrier[team]
            if c is not None and last_carrier[team] is None:
                pickups[1 - team] += 1
            last_carrier[team] = c
        if w.t >= next_f:
            next_f += 1.0 / FPS
            img = Image.fromarray(np.ascontiguousarray(cam.render()[0][..., :3]).astype(np.uint8))
            dr = ImageDraw.Draw(img)
            dr.rectangle([0, 0, W, 88], fill=(255, 255, 255))
            dr.text((20, 8), a.title, font=fb, fill=(20, 20, 30))
            tags = list(sc.tags)
            dr.text((20, 52), "%s (azul) %d  x  %d %s (vermelho)   |   pegadas %d x %d   segurados %d x %d"
                    % (names[0], sc.score[0], sc.score[1], names[1], pickups[0], pickups[1],
                       tags[0] + tags[1], tags[2] + tags[3]), font=fs, fill=(40, 40, 60))
            dr.text((W - 150, 52), "%5.1f s" % w.t, font=fs, fill=(40, 40, 60))
            if w.message:
                tw = dr.textlength(w.message, font=fm)
                dr.rectangle([(W - tw) / 2 - 16, H - 80, (W + tw) / 2 + 16, H - 30], fill=(150, 10, 33))
                dr.text(((W - tw) / 2, H - 76), w.message, font=fm, fill=(255, 255, 255))
            ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()
    print("saved %s: %s %d x %d %s (%.0f s)" % (out, names[0], sc.score[0], sc.score[1], names[1], w.t))


if __name__ == "__main__":
    main()
