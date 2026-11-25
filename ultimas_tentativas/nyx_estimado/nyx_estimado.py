#!/usr/bin/env python3
"""Rouba-bandeira: estimativa do comportamento do time "Grupo 3 · Nyx", reconstruída a partir das
gravações ao vivo do torneio (engenharia reversa pelos trajetos; o código deles não foi visto).
Serve como adversário de treino/teste. Análise: ultimas_tentativas/nyx_estimado/README.md

Regras observadas (referencial do time: base própria em (-0.62, 0), ataca +x):
  DEFENSOR
    * fica parado EM CIMA da própria bandeira (mediana 4 mm da base) virado para o atacante inimigo;
    * só sai quando a bandeira é roubada (381 de 397 saídas): persegue quem a carrega, direto e na
      velocidade máxima, enquanto o carregador está no próprio campo (x < 0); no meio do campo desiste;
    * volta para a base de frente (gira e anda), e ao chegar vira para o atacante inimigo.
  ATACANTE
    * vai direto para a bandeira inimiga na velocidade máxima (0,20 m/s);
    * não entra no raio de ~0,13 m de um defensor inimigo ativo (preso = 0,10 m): contorna, e se a
      bandeira está "coberta" pelo defensor fica rodeando a ~0,14 m esperando uma brecha;
    * com a bandeira: vai para casa (contornando defensores do mesmo jeito);
    * se for preso perto da bandeira inimiga: depois de solto vai para o ponto de espera
      (-0,10; ±0,33), no lado oposto ao que estava, espera ~10 s e tenta de novo.

Mesmo formato do template do kit (kit/exercicios/ctf/ctf.py): um nó por robô, ~role escolhe o papel.
Assina também /game/state (presos, quem carrega). Só Python padrão, rospy e psr_hero.utils.
    ./run.sh run ctf blue:=nyx_estimado.py red:=...
"""
import json
import math

import rospy
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String

from psr_hero.utils import V_MAX, W_MAX, clamp, pose_from_odom, wrap_angle

BASE_X = 0.62
KEEP_OUT = 0.13  # m: o atacante não chega mais perto que isso de um defensor inimigo ativo
ORBIT_R = 0.165  # m: raio da órbita em volta do defensor que cobre a bandeira (v/r ~ 0,2/0,165 = 1,2 rad/s)
ORBIT_ZONE = 0.20  # m: a partir desta distância do defensor passa a orbitar
K_R = 6.0  # correção radial da órbita
WAIT_POINT = (-0.10, 0.33)  # ponto de espera (y com o sinal oposto ao do lugar onde foi preso)
WAIT_S = 10.0  # s de espera depois de ser preso perto da bandeira inimiga
CHASE_LIMIT_X = -0.03  # o defensor persegue o carregador só enquanto ele está antes disto (próprio campo)
HOME_TOL = 0.012  # m
K_W = 4.0  # ganho do giro (rad/s por rad)


class CaptureTheFlag:
    def __init__(self):
        rospy.init_node("ctf")
        self.robot = rospy.get_param("~robot", 0)
        self.team = rospy.get_param("~team", "blue" if self.robot == 0 else "red")
        self.role = rospy.get_param("~role", "attacker")
        self.opponents = list(rospy.get_param("~opponents", [1 if self.robot == 0 else 0]))
        self.blue = self.team == "blue"
        self.s = 1.0 if self.blue else -1.0  # referencial do time: (x, y) -> (s x, s y)
        self.me = None
        self.opp = {}
        self.enemy_flag = None
        self.own_flag = None
        self.carrying = False
        self.game = {}
        self.now = 0.0
        # estado do atacante
        self.was_held = False
        self.held_y = 0.0
        self.held_x = 0.0
        self.mode = "go"
        self.wait_until = None
        self.pub = rospy.Publisher("/hero_%d/cmd_vel" % self.robot, Twist, queue_size=1)
        rospy.Subscriber("/hero_%d/odom" % self.robot, Odometry, self.on_me)
        for k in self.opponents:
            rospy.Subscriber("/hero_%d/odom" % k, Odometry, self.on_opp, callback_args=k)
        rospy.Subscriber("/flag/%s" % ("red" if self.blue else "blue"), PointStamped, self.on_enemy_flag)
        rospy.Subscriber("/flag/%s" % self.team, PointStamped, self.on_own_flag)
        rospy.Subscriber("/hero_%d/has_flag" % self.robot, Bool, self.on_has_flag)
        rospy.Subscriber("/game/state", String, self.on_game)

    # --- ROS -------------------------------------------------------------------------------
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

    # --- helpers ---------------------------------------------------------------------------
    def tf(self, p):
        """(x, y[, yaw]) do mundo -> referencial do time."""
        if len(p) == 3:
            return (self.s * p[0], self.s * p[1], wrap_angle(p[2] + (0.0 if self.blue else math.pi)))
        return (self.s * p[0], self.s * p[1])

    def held(self, k):
        h = self.game.get("held") or []
        return bool(h[k]) if k < len(h) else False

    def carrying_of(self, k):
        c = self.game.get("carrying") or []
        return bool(c[k]) if k < len(c) else False

    @staticmethod
    def steer(me, dx, dy, v_max=V_MAX, stop_dist=None):
        """Vai na direção (dx, dy): gira no lugar se o erro passar de 90 graus, senão anda e corrige."""
        if abs(dx) + abs(dy) < 1e-9:
            return 0.0, 0.0
        err = wrap_angle(math.atan2(dy, dx) - me[2])
        w = clamp(K_W * err, -W_MAX, W_MAX)
        if abs(err) > math.pi / 2:
            return 0.0, w
        v = v_max * math.cos(err)
        if stop_dist is not None:
            v = min(v, 2.5 * stop_dist)
        return v, w

    # --- roles -----------------------------------------------------------------------------
    def compute_command(self, me, opponents, enemy_flag, own_flag, carrying):
        me = self.tf(me)
        opps = {k: self.tf(p) for k, p in self.opp.items()}
        ef, of = self.tf(enemy_flag), self.tf(own_flag)
        if self.role == "defender":
            return self.defend(me, opps, of)
        return self.attack(me, opps, ef, carrying)

    def defend(self, me, opps, of):
        home = (-BASE_X, 0.0)
        taken = math.hypot(of[0] - home[0], of[1] - home[1]) > 0.03
        if taken and of[0] < CHASE_LIMIT_X:
            # persegue a própria bandeira (= quem a carrega), direto
            return self.steer(me, of[0] - me[0], of[1] - me[1])
        d = math.hypot(home[0] - me[0], home[1] - me[1])
        if d > HOME_TOL:
            return self.steer(me, home[0] - me[0], home[1] - me[1], stop_dist=d)
        # na base: vira para o atacante inimigo mais avançado
        if not opps:
            return 0.0, 0.0
        k = min(opps, key=lambda j: opps[j][0])
        err = wrap_angle(math.atan2(opps[k][1] - me[1], opps[k][0] - me[0]) - me[2])
        return 0.0, clamp(K_W * err, -W_MAX, W_MAX) if abs(err) > 0.08 else 0.0

    def attack(self, me, opps, ef, carrying):
        held = self.held(self.robot)
        if held and not self.was_held:
            self.held_x, self.held_y = me[0], me[1]
        if self.was_held and not held and self.held_x > 0.3 and not carrying:
            self.mode, self.wait_until = "wait", None  # foi preso perto da bandeira: vai esperar
        self.was_held = held
        if held:
            return 0.0, 0.0
        if carrying:
            self.mode = "go"
        if self.mode == "wait":
            wp = (WAIT_POINT[0], -math.copysign(WAIT_POINT[1], self.held_y or 1.0))
            d = math.hypot(wp[0] - me[0], wp[1] - me[1])
            if d > 0.03 and self.wait_until is None:
                return self.steer(me, wp[0] - me[0], wp[1] - me[1], stop_dist=d)
            if self.wait_until is None:
                self.wait_until = self.now + WAIT_S
            if self.now < self.wait_until:
                return 0.0, 0.0
            self.mode = "go"
        goal = (-BASE_X, 0.0) if carrying else ef
        gx, gy = goal[0] - me[0], goal[1] - me[1]
        n = math.hypot(gx, gy) or 1.0
        gx, gy = gx / n, gy / n
        # defensores inimigos ativos (soltos, no campo deles): órbita a ORBIT_R com o alvo à esquerda/direita
        best = None
        for k, o in opps.items():
            if self.held(k) or o[0] <= 0.0:
                continue
            d = math.hypot(me[0] - o[0], me[1] - o[1])
            if d < ORBIT_ZONE and (best is None or d < best[0]):
                best = (d, o)
        if best is not None:
            d, o = best
            ux, uy = (me[0] - o[0]) / max(d, 1e-6), (me[1] - o[1]) / max(d, 1e-6)
            tx, ty = -uy, ux  # tangente (sentido anti-horário em volta do defensor)
            if tx * gx + ty * gy < 0:  # sentido que leva para o lado do alvo
                tx, ty = -tx, -ty
            radial = K_R * (ORBIT_R - d)  # > 0: muito perto, sai; < 0: longe, aproxima
            if d < KEEP_OUT:
                radial = max(radial, 1.5)
            gx, gy = tx + radial * ux, ty + radial * uy
        return self.steer(me, gx, gy)

    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            self.now = rospy.get_time()
            if (self.me is None or len(self.opp) < len(self.opponents)
                    or self.enemy_flag is None or self.own_flag is None):
                rospy.logwarn_throttle(2.0, "esperando odometria e bandeiras")
            else:
                v, w = self.compute_command(self.me, list(self.opp.values()), self.enemy_flag,
                                            self.own_flag, self.carrying)
                cmd = Twist()
                cmd.linear.x, cmd.angular.z = v, w
                self.pub.publish(cmd)
            rate.sleep()


if __name__ == "__main__":
    CaptureTheFlag().run()
