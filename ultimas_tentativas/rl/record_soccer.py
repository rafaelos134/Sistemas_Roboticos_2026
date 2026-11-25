"""Record a soccer match of two kit nodes in psr_sim as MP4 (offscreen camera), like record_video.py.

    $PY record_soccer.py --blue exports/soccer/soccer_rl.py --red exports/soccer/soccer_script.py \
        --seconds 90 --title "..." --out docs/videos/futebol_x.mp4
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from eval_soccer import TeamDriver
from record_video import FPS, font

W, H = 1280, 720


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--blue", required=True)
    p.add_argument("--red", default=None)
    p.add_argument("--seconds", type=float, default=90)
    p.add_argument("--title", default="")
    p.add_argument("--seed", type=int, default=3)
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
    drivers = [TeamDriver(a.blue, "blue", "blue_node")] + ([TeamDriver(a.red, "red", "red_node")] if a.red else [])
    names = [drivers[0].name, drivers[1].name if a.red else "parado"]
    sc = SCENARIOS["soccer"](6, a.seed)
    w = World(sc, backend="gpu")
    cam = cams[0]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H),
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset",
                           "veryfast", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    fb, fs, fm = font(32), font(20), font(26)
    next_t = next_f = 0.0
    state = w.read()
    while w.t < a.seconds:
        if w.pending is None and w.t >= next_t:
            for d in drivers:
                for i, (v, om) in d.act(state["robots"], state["puck"], w.t):
                    w.set_cmd(i, v, om)
            next_t = w.t + 0.05
        was = w.pending is not None
        state = w.tick()
        if was and w.pending is None:
            for d in drivers:
                d.new_kickoff()
        if w.t >= next_f:
            next_f += 1.0 / FPS
            img = Image.fromarray(np.ascontiguousarray(cam.render()[0][..., :3]).astype(np.uint8))
            dr = ImageDraw.Draw(img)
            dr.rectangle([0, 0, W, 88], fill=(255, 255, 255))
            dr.text((20, 8), a.title, font=fb, fill=(20, 20, 30))
            dr.text((20, 52), "%s (azul) %d  x  %d %s (vermelho)" % (names[0], sc.score[0], sc.score[1], names[1]),
                    font=fs, fill=(40, 40, 60))
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
