#!/usr/bin/env python3
"""Rouba-bandeira: máquina de estados estimada da equipe Corsa (engenharia reversa das gravações ao vivo
do torneio; o código da equipe não foi visto). Formato do template do kit (exercicios/ctf/ctf.py): um nó por
robô, ~role escolhe o papel. Assina também /game/state (quem está preso).

Referencial do time: base própria em (-0.62, 0), ataca +x. "Invasor" = adversário mais avançado (menor x),
"defensor inimigo" = o outro.

DEFENSOR (PARAMS["def"])
  PERSEGUE  invasor (solto; CH_HELD = 1: também preso, continua empurrando; CH_HELD = 2: preso, fico parado) no meu campo (x < CH_X) e [bandeira roubada, ou CH_ALWAYS, ou invasor a menos de
            CH_R da minha base]: vai direto nele (ou na minha bandeira, se ela foi roubada)
  GUARDA    senão: G_MODE 0 = ponto a G_DIST da base na direção do invasor; 1 = linha x = -0.62 + G_DIST,
            y = G_YK * y do invasor (limitado a ±G_YL). Chega devagar e, parado, vira para o invasor (FACE)
ATACANTE (PARAMS["att"])
  ESPERA    se foi preso com x > WAIT_X: nos WAIT_S s depois de solto vai para (WAIT_PX, ±WAIT_PY) (lado
            oposto ao que estava) e fica lá
  VOLTA     com a bandeira: vai para casa (C_REV = 1: sempre de ré), velocidade C_V
  VAI       senão: vai para a bandeira inimiga
  (VOLTA e VAI) desvio: de cada adversário solto no campo dele a menos de AV_R soma uma repulsão AV_K e
            um giro tangente AV_T (contorna)
Direção (os dois): gira no lugar se o erro passar de TURN, anda com V * cos(erro); REV = 1 anda de ré
quando o alvo está atrás. Parâmetros ajustados aos comandos deduzidos das gravações (eng_reversa/ajustar.py).
"""
import json
import math

import numpy as np
import rospy
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String

from psr_hero.utils import V_MAX, W_MAX, pose_from_odom

BASE_X = 0.62
PARAMS = {'att': {'WAIT_X': 9.0,
         'WAIT_S': 6.0,
         'WAIT_PX': 0.1,
         'WAIT_PY': 0.5,
         'AV_R': 0.3,
         'AV_K': 2.0,
         'AV_T': 0.5,
         'K_W': 0.5,
         'TURN': 3.2,
         'V': 1.0,
         'REV': 1,
         'C_REV': 0,
         'C_V': 1.0},
 'def': {'CH_X': 0.0,
         'CH_ALWAYS': 0,
         'CH_R': 0.0,
         'CH_HELD': 0,
         'G_MODE': 1,
         'G_DIST': 0.57,
         'G_YK': 0.0,
         'G_YL': 0.35,
         'FACE': 0,
         'K_W': 0.5,
         'TURN': 3.2,
         'V': 1.0,
         'REV': 1,
         'K_D': 1.0}}

ESPACO_DEF = {
    "CH_X": [-0.3, -0.15, 0.0, 0.1],
    "CH_ALWAYS": [0, 1],
    "CH_R": [0.0, 0.2, 0.3, 0.45, 0.6],
    "CH_HELD": [0, 1, 2],
    "G_MODE": [0, 1],
    "G_DIST": [0.0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.57],
    "G_YK": [0.0, 0.5, 1.0],
    "G_YL": [0.1, 0.2, 0.35],
    "FACE": [0, 1],
    "K_W": [0.5, 1.0, 2.0, 4.0, 8.0],
    "TURN": [0.3, 0.6, 1.0, 1.57, 3.2],
    "V": [0.5, 0.8, 1.0],
    "REV": [0, 1],
    "K_D": [1.0, 2.5, 5.0],
}
PADRAO_DEF = {"CH_X": 0.0, "CH_ALWAYS": 0, "CH_R": 0.3, "CH_HELD": 0, "G_MODE": 0, "G_DIST": 0.0, "G_YK": 0.5, "G_YL": 0.2,
              "FACE": 1, "K_W": 4.0, "TURN": 1.57, "V": 1.0, "REV": 0, "K_D": 2.5}
ESPACO_ATT = {
    "WAIT_X": [0.3, 0.45, 9.0],
    "WAIT_S": [3.0, 6.0, 10.0, 14.0],
    "WAIT_PX": [-0.3, -0.1, 0.1],
    "WAIT_PY": [0.2, 0.33, 0.5],
    "AV_R": [0.0, 0.12, 0.16, 0.2, 0.25, 0.3],
    "AV_K": [0.5, 1.0, 2.0, 4.0],
    "AV_T": [0.0, 0.5, 1.0, 2.0],
    "K_W": [0.5, 1.0, 2.0, 4.0, 8.0],
    "TURN": [0.3, 0.6, 1.0, 1.57, 3.2],
    "V": [0.5, 0.8, 1.0],
    "REV": [0, 1],
    "C_REV": [0, 1],
    "C_V": [0.5, 0.8, 1.0],
}
PADRAO_ATT = {"WAIT_X": 9.0, "WAIT_S": 6.0, "WAIT_PX": -0.1, "WAIT_PY": 0.33, "AV_R": 0.0, "AV_K": 1.0,
              "AV_T": 0.0, "K_W": 4.0, "TURN": 1.57, "V": 1.0, "REV": 0,
              "C_REV": 0, "C_V": 1.0}


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _steer(x, y, yaw, dx, dy, P, dist=None):
    e = _wrap(np.arctan2(dy, dx) - yaw)
    sg = np.ones_like(e)
    if P["REV"]:
        tras = np.abs(e) > np.pi / 2
        sg = np.where(tras, -1.0, 1.0)
        e = np.where(tras, _wrap(e + np.pi), e)
    w = np.clip(P["K_W"] * e, -W_MAX, W_MAX)
    v = np.where(np.abs(e) > P["TURN"], 0.0, P["V"] * V_MAX * np.maximum(np.cos(e), 0.0))
    if dist is not None:
        v = np.minimum(v, P.get("K_D", 2.5) * dist)
    return v * sg, w


def politica_def(F, P):
    """F (referencial do time): x y yaw, o1x o1y (invasor), o1h (preso), ofx ofy (minha bandeira)."""
    x, y, yaw = F["x"], F["y"], F["yaw"]
    o1x, o1y = F["o1x"], F["o1y"]
    hx, hy = -BASE_X, 0.0
    taken = np.hypot(F["ofx"] - hx, F["ofy"] - hy) > 0.03
    d1 = np.hypot(o1x - hx, o1y - hy)
    chase = (o1x < P["CH_X"]) & (~F["o1h"] | (P["CH_HELD"] == 1)) & (taken | bool(P["CH_ALWAYS"]) | (d1 < P["CH_R"]))
    tx = np.where(taken, F["ofx"], o1x)
    ty = np.where(taken, F["ofy"], o1y)
    vc, wc = _steer(x, y, yaw, tx - x, ty - y, P)
    if P["G_MODE"] == 0:
        n = np.maximum(d1, 1e-6)
        gx, gy = hx + P["G_DIST"] * (o1x - hx) / n, hy + P["G_DIST"] * (o1y - hy) / n
    else:
        gx, gy = hx + P["G_DIST"] + 0 * x, np.clip(P["G_YK"] * o1y, -P["G_YL"], P["G_YL"])
    dg = np.hypot(gx - x, gy - y)
    vg, wg = _steer(x, y, yaw, gx - x, gy - y, P, dist=dg)
    parado = dg < 0.02
    if P["FACE"]:
        e = _wrap(np.arctan2(o1y - y, o1x - x) - yaw)
        vg = np.where(parado, 0.0, vg)
        wg = np.where(parado, np.where(np.abs(e) > 0.08, np.clip(P["K_W"] * e, -W_MAX, W_MAX), 0.0), wg)
    else:
        vg, wg = np.where(parado, 0.0, vg), np.where(parado, 0.0, wg)
    v, w = np.where(chase, vc, vg), np.where(chase, wc, wg)
    if P["CH_HELD"] == 2:  # invasor preso no meu campo: fico parado onde estou
        fica = F["o1h"] & (o1x < P["CH_X"]) & ~taken
        v, w = np.where(fica, 0.0, v), np.where(fica, 0.0, w)
    return v, w


def politica_att(F, P):
    """F: x y yaw, efx efy (bandeira inimiga), carry, o1x o1y o1h, o2x o2y o2h, t_rel (s desde que fui solto),
    hx_prev hy_prev (onde fui preso)."""
    x, y, yaw = F["x"], F["y"], F["yaw"]
    gx = np.where(F["carry"], -BASE_X, F["efx"]) - x
    gy = np.where(F["carry"], 0.0, F["efy"]) - y
    n = np.maximum(np.hypot(gx, gy), 1e-6)
    gx, gy = gx / n, gy / n
    if P["AV_R"] > 0:
        ax, ay = gx.copy(), gy.copy()
        for k in ("o1", "o2"):
            ox, oy = F[k + "x"], F[k + "y"]
            d = np.maximum(np.hypot(x - ox, y - oy), 1e-6)
            ativo = (~F[k + "h"]) & (ox > 0) & (d < P["AV_R"])
            ux, uy = (x - ox) / d, (y - oy) / d
            tx, ty = -uy, ux
            sgn = np.where(tx * gx + ty * gy < 0, -1.0, 1.0)
            f = P["AV_K"] * (P["AV_R"] - d) / P["AV_R"]
            ax = np.where(ativo, ax + f * ux + P["AV_T"] * sgn * tx, ax)
            ay = np.where(ativo, ay + f * uy + P["AV_T"] * sgn * ty, ay)
        gx, gy = ax, ay
    v, w = _steer(x, y, yaw, gx, gy, P)
    # com a bandeira: C_REV = 1 volta sempre de ré (de costas para casa), com velocidade C_V
    Pc = dict(P, V=P["C_V"])
    if P["C_REV"]:
        e = _wrap(np.arctan2(gy, gx) - yaw - np.pi)
        wc = np.clip(P["K_W"] * e, -W_MAX, W_MAX)
        vc = -np.where(np.abs(e) > P["TURN"], 0.0, P["C_V"] * V_MAX * np.maximum(np.cos(e), 0.0))
    else:
        vc, wc = _steer(x, y, yaw, gx, gy, Pc)
    v, w = np.where(F["carry"], vc, v), np.where(F["carry"], wc, w)
    esp = (F["t_rel"] < P["WAIT_S"]) & (F["hx_prev"] > P["WAIT_X"]) & ~F["carry"]
    wy = -np.sign(np.where(F["hy_prev"] == 0, 1.0, F["hy_prev"])) * P["WAIT_PY"]
    dw = np.hypot(P["WAIT_PX"] - x, wy - y)
    vw, ww = _steer(x, y, yaw, P["WAIT_PX"] - x, wy - y, P, dist=dw)
    vw, ww = np.where(dw < 0.03, 0.0, vw), np.where(dw < 0.03, 0.0, ww)
    return np.where(esp, vw, v), np.where(esp, ww, w)


FAMILIAS = {"def": (politica_def, ESPACO_DEF, PADRAO_DEF), "att": (politica_att, ESPACO_ATT, PADRAO_ATT)}


class CaptureTheFlag:
    def __init__(self):
        rospy.init_node("ctf")
        self.robot = rospy.get_param("~robot", 0)
        self.team = rospy.get_param("~team", "blue" if self.robot == 0 else "red")
        self.role = rospy.get_param("~role", "attacker")
        self.opponents = list(rospy.get_param("~opponents", [1 if self.robot == 0 else 0]))
        self.s = 1.0 if self.team == "blue" else -1.0
        self.me, self.opp, self.enemy_flag, self.own_flag, self.carrying = None, {}, None, None, False
        self.game = {}
        self.was_held, self.t_release, self.held_at = False, -1e9, (0.0, 0.0)
        self.now = 0.0
        self.pub = rospy.Publisher("/hero_%d/cmd_vel" % self.robot, Twist, queue_size=1)
        rospy.Subscriber("/hero_%d/odom" % self.robot, Odometry, self.on_me)
        for k in self.opponents:
            rospy.Subscriber("/hero_%d/odom" % k, Odometry, self.on_opp, callback_args=k)
        rospy.Subscriber("/flag/%s" % ("red" if self.team == "blue" else "blue"), PointStamped, self.on_enemy_flag)
        rospy.Subscriber("/flag/%s" % self.team, PointStamped, self.on_own_flag)
        rospy.Subscriber("/hero_%d/has_flag" % self.robot, Bool, self.on_has_flag)
        rospy.Subscriber("/game/state", String, self.on_game)

    def on_me(self, msg):
        self.me = pose_from_odom(msg)

    def on_opp(self, msg, k):
        self.opp[k] = pose_from_odom(msg)

    def on_enemy_flag(self, msg):
        self.enemy_flag = (msg.point.x, msg.point.y)

    def on_own_flag(self, msg):
        self.own_flag = (msg.point.x, msg.point.y)

    def on_has_flag(self, msg):
        self.carrying = msg.data

    def on_game(self, msg):
        try:
            self.game = json.loads(msg.data)
        except ValueError:
            pass

    def is_held(self, k):
        h = self.game.get("held") or []
        return bool(h[k]) if k < len(h) else False

    def tf(self, p):
        if len(p) == 3:
            return (self.s * p[0], self.s * p[1], math.atan2(math.sin(p[2] + (0 if self.s > 0 else math.pi)),
                                                             math.cos(p[2] + (0 if self.s > 0 else math.pi))))
        return (self.s * p[0], self.s * p[1])

    def compute_command(self, me, opponents, enemy_flag, own_flag, carrying):
        me = self.tf(me)
        held = self.is_held(self.robot)
        if held and not self.was_held:
            self.held_at = (me[0], me[1])
        if self.was_held and not held:
            self.t_release = self.now
        self.was_held = held
        opp = sorted(((self.tf(p), self.is_held(k)) for k, p in self.opp.items()), key=lambda q: q[0][0])
        o1, o2 = opp[0], opp[-1]
        ef, of = self.tf(enemy_flag), self.tf(own_flag)
        A = lambda val: np.array([val])  # noqa: E731
        F = {"x": A(me[0]), "y": A(me[1]), "yaw": A(me[2]), "o1x": A(o1[0][0]), "o1y": A(o1[0][1]), "o1h": A(o1[1]),
             "o2x": A(o2[0][0]), "o2y": A(o2[0][1]), "o2h": A(o2[1]), "efx": A(ef[0]), "efy": A(ef[1]),
             "ofx": A(of[0]), "ofy": A(of[1]), "carry": A(bool(carrying)), "t_rel": A(self.now - self.t_release),
             "hx_prev": A(self.held_at[0]), "hy_prev": A(self.held_at[1])}
        papel = "def" if self.role == "defender" else "att"
        v, w = FAMILIAS[papel][0](F, PARAMS[papel])
        return float(v[0]), float(w[0])

    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            self.now = rospy.get_time()
            if (self.me is None or len(self.opp) < len(self.opponents)
                    or self.enemy_flag is None or self.own_flag is None):
                rospy.logwarn_throttle(2.0, "waiting for odometry and flags")
            else:
                v, w = self.compute_command(self.me, list(self.opp.values()), self.enemy_flag,
                                            self.own_flag, self.carrying)
                cmd = Twist()
                cmd.linear.x, cmd.angular.z = v, w
                self.pub.publish(cmd)
            rate.sleep()


if __name__ == "__main__":
    CaptureTheFlag().run()
