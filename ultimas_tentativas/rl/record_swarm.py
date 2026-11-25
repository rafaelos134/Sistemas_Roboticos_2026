"""3D video of the sumo training: hundreds of rings simulated at the same time (the vectorized
Genesis environment of train_sumo.py), filmed by one moving camera.

    ~/ros2_ws/.venv/bin/python record_swarm.py --out runs/swarm.mp4            # full video
    ~/ros2_ws/.venv/bin/python record_swarm.py --test --out runs/swarm_test.png # one frame per camera pose

Three segments, each with its own policy, all with sampled (exploring) actions as in training:
  1. untrained network vs a stopped robot          ("a rede ainda não sabe nada")
  2. phase 1 policy (sumo_ppo_v1) vs a stopped robot ("aprendeu a empurrar")
  3. phase 5 policy (sumo_ppo_v5) vs the scripted opponents ("treino final, rivais que se mexem")
The rings are drawn only (no collision): the training environment ends a round when a centre
crosses r = 0.40 m and restarts that ring at once.
"""
import argparse
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import genesis as gs  # noqa: F401  (imported by sumo_env too)

W, H, FPS = 1280, 720, 20
SPACING = 1.0
cams = []
_build = gs.Scene.build


def font(size):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def build_with_rings(scene, *a, n_envs=1, **k):
    """Add the floor, the drawn rings and the camera, then build with the envs side by side."""
    scene.add_entity(gs.morphs.Plane(pos=(0, 0, -0.03), collision=False),
                     surface=gs.surfaces.Default(color=(0.80, 0.80, 0.82, 1.0)))
    scene.add_entity(gs.morphs.Cylinder(radius=0.412, height=0.012, pos=(0, 0, -0.0075), fixed=True, collision=False),
                     surface=gs.surfaces.Default(color=(0.96, 0.96, 0.96, 1.0)))
    scene.add_entity(gs.morphs.Cylinder(radius=0.40, height=0.012, pos=(0, 0, -0.0065), fixed=True, collision=False),
                     surface=gs.surfaces.Default(color=(0.12, 0.12, 0.14, 1.0)))
    cams.append(scene.add_camera(res=(W, H), pos=(0, -3, 3), lookat=(0, 0, 0), fov=45, near=0.05, far=200.0, GUI=False))
    return _build(scene, *a, n_envs=n_envs, env_spacing=(SPACING, SPACING), **k)


gs.Scene.build = build_with_rings

from eval_vec import load  # noqa: E402
from ppo import ActorCritic  # noqa: E402
from sumo_env import RewardConfig, SumoEnv  # noqa: E402
from sumo_obs import OBS_DIM  # noqa: E402

SEGMENTS = [
    dict(policy=None, mix={"stopped": 1.0}, seconds=9,
         title="Início do treino", caption="a rede ainda não sabe nada: ações aleatórias"),
    dict(policy="runs/sumo_ppo_v1/model.pt", mix={"stopped": 1.0}, seconds=9,
         title="Fase 1 · 2,5 milhões de passos depois", caption="aprendeu a girar e empurrar o robô parado"),
    dict(policy="runs/sumo_ppo_v5/pool/it1200.pt",
         mix={"pursuer": 0.25, "pursuer_bi": 0.2, "chaser": 0.2, "careful": 0.15, "evader": 0.1, "random": 0.1},
         seconds=14, title="Fase 5 · treino final", caption="contra rivais que se mexem e perseguem"),
]


def grid_extent(n):
    side = int(math.ceil(math.sqrt(n)))
    return (side - 1) * SPACING / 2  # half size of the grid of ring centres


CENTER = [0.0, 0.0]
NEAR = [0.0, 0.0]  # a ring near the centre of the grid


def camera_pose(seg, u, half):
    """Camera (pos, lookat) at progress u in [0, 1] of segment seg, around the grid centre."""
    pos, look = _camera_pose(seg, u, half)
    cx, cy = CENTER
    if seg == 2:  # the dive ends on one ring
        cx = CENTER[0] + (NEAR[0] - CENTER[0]) * (0.5 - 0.5 * math.cos(math.pi * u))
        cy = CENTER[1] + (NEAR[1] - CENTER[1]) * (0.5 - 0.5 * math.cos(math.pi * u))
    return (pos[0] + cx, pos[1] + cy, pos[2]), (look[0] + cx, look[1] + cy, look[2])


def _camera_pose(seg, u, half):
    s = lambda a, b: a + (b - a) * (0.5 - 0.5 * math.cos(math.pi * u))  # noqa: E731  (ease in/out)
    if seg == 0:  # start close on a few rings, pull back to reveal the whole grid
        d = s(1.6, 1.3 * half + 1.5)
        ang = s(-0.35, 0.15)
        return (d * math.sin(ang), -d * math.cos(ang), s(1.1, 1.5 * half + 2.0)), (0, 0, 0)
    if seg == 1:  # slow orbit over the grid
        ang = 0.15 + 0.6 * u
        d = 1.3 * half + 1.5
        return (d * math.sin(ang), -d * math.cos(ang), 1.5 * half + 2.0), (0, 0, 0)
    # seg 2: from wide, dive into one ring near the centre
    d = s(1.3 * half + 1.5, 0.95)
    ang = s(0.75, 1.1)
    return (d * math.sin(ang), -d * math.cos(ang), s(1.5 * half + 2.0, 0.75)), (0, 0, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int, default=400)
    p.add_argument("--out", required=True)
    p.add_argument("--test", action="store_true")
    args = p.parse_args()

    env = SumoEnv(n_envs=args.envs, reward=RewardConfig(draw=-3.0), random_start=0.3, obs_noise=0.0)
    cam = cams[0]
    half = grid_extent(args.envs)
    off = np.asarray(env.scene.envs_offset)
    off = off[:, :2] - off[0, :2]  # the camera pose is expressed relative to env 0
    CENTER[:] = off.mean(0).tolist()
    k = int(np.argmin(np.hypot(off[:, 0] - CENTER[0], off[:, 1] - CENTER[1])))
    NEAR[:] = off[k].tolist()
    print("grid centre", CENTER, "near ring", NEAR)
    f_big, f_mid, f_small = font(38), font(26), font(22)

    if args.test:
        imgs = []
        for seg in range(3):
            for u in (0.0, 1.0):
                pos, look = camera_pose(seg, u, half)
                cam.set_pose(pos=pos, lookat=look)
                imgs.append(Image.fromarray(np.ascontiguousarray(cam.render()[0][..., :3]).astype(np.uint8)).resize((640, 360)))
        sheet = Image.new("RGB", (1280, 1080))
        for k, im in enumerate(imgs):
            sheet.paste(im, ((k % 2) * 640, (k // 2) * 360))
        sheet.save(args.out)
        print("saved", args.out)
        return

    out = Path(args.out)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-crf", "22", "-preset", "veryfast", "-movflags", "+faststart", str(out)],
                          stdin=subprocess.PIPE)
    torch.manual_seed(0)
    for si, seg in enumerate(SEGMENTS):
        if seg["policy"] is None:
            model = ActorCritic(OBS_DIM, 2).cuda().eval()
            with torch.no_grad():
                model.log_std.fill_(math.log(0.6))
        else:
            model = load(seg["policy"])
            with torch.no_grad():
                model.log_std.fill_(math.log(0.2))  # exploring, as during training
        env.set_opponent_mix(seg["mix"])
        obs = env.reset_all()
        wins = rounds = 0
        n = seg["seconds"] * FPS
        for k in range(n):
            with torch.no_grad():
                act = model.dist(model.norm(obs)).sample()
            obs, _, term, trunc, info = env.step(act)
            done = term | trunc
            rounds += int(done.sum())
            wins += int((info["win"] & done).sum())
            pos, look = camera_pose(si, k / (n - 1), half)
            cam.set_pose(pos=pos, lookat=look)
            img = Image.fromarray(np.ascontiguousarray(cam.render()[0][..., :3]).astype(np.uint8))
            d = ImageDraw.Draw(img, "RGBA")
            d.rectangle([0, 0, W, 104], fill=(255, 255, 255, 225))
            d.text((24, 12), seg["title"], font=f_big, fill=(15, 15, 25))
            d.text((24, 62), seg["caption"], font=f_mid, fill=(60, 60, 75))
            stats = "%d ringues ao mesmo tempo" % args.envs
            d.text((W - 24 - d.textlength(stats, font=f_small), 18), stats, font=f_small, fill=(42, 120, 214))
            if rounds:
                st2 = "rodadas: %d   vitórias: %d%%" % (rounds, round(100 * wins / rounds))
                d.text((W - 24 - d.textlength(st2, font=f_small), 60), st2, font=f_small, fill=(60, 60, 75))
            ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()
    print("saved", out)


if __name__ == "__main__":
    main()
