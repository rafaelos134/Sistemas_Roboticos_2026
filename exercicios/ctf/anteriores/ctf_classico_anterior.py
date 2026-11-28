#!/usr/bin/env python3
"""Capture the flag (student template). One file drives every robot of your team.

The arena is 1.6 x 1.6 m, split at x = 0. Blue (base on the left, x = -0.62 m) has
hero_0 and hero_1; red (base on the right, x = +0.62 m) has hero_2 and hero_3.
(With two robots: hero_0 is blue and hero_1 is red.) In a team of two, one robot
is the attacker and the other the defender (parameter ~role): the attacker tries to
steal the flag while the defender blocks.

Rules
  * Touch the enemy flag (< 0.07 m) and you carry it. It follows you.
  * Carry it into your base circle (radius 0.14 m around your flag) to score a point.
  * Inside YOUR half, touching an opponent (< 0.10 m) HOLDS it still for 4 s (its commands
    are ignored; the game goes on for everybody else). If it was carrying your flag, the
    flag returns to your base. A held robot cannot hold others, and after the release it
    is immune for 1.5 s.
  * In the opponents' half you are the one who can be held: block one opponent while
    your teammate goes for the flag.
  * 90 s without a point is a draw and everyone goes back to the start.

Topics
  /hero_N/odom        nav_msgs/Odometry          pose of every robot
  /hero_N/has_flag    std_msgs/Bool              robot N carries the enemy flag
  /flag/blue          geometry_msgs/PointStamped position of the blue flag
  /flag/red           geometry_msgs/PointStamped position of the red flag
  /hero_N/laser       sensor_msgs/LaserScan      the 8-beam lidar also sees the other robots
  /hero_N/cmd_vel     geometry_msgs/Twist        your command

Parameters: ~robot (your id), ~team ('blue' or 'red'), ~role ('attacker' or 'defender'),
~opponents (list of ids). The launch file fills them in.

Run:  ./run.sh run ctf      (see README.md in this folder)
"""
import json
import math  # noqa: F401

import rospy
from geometry_msgs.msg import PointStamped, Twist  # noqa: F401
from nav_msgs.msg import Odometry  # noqa: F401
from std_msgs.msg import Bool, String  # noqa: F401

from psr_hero.utils import V_MAX, W_MAX, clamp, pose_from_odom, wrap_angle  # noqa: F401

BASE_X = 0.62  # m, blue at -BASE_X, red at +BASE_X

ARENA_HALF = 0.80   # paredes em x, y = ±0,8 m
AVOID_R = 0.30      # começa a desviar com o inimigo a 30 cm
K_REP = 1.0         # ganho da repulsão
K_TAN = 2.0         # peso da componente tangencial
K_RHO = 3.0         # m/s of forward speed per metre of distance to the goal
K_ALPHA = 6.0       # rad/s of turn per radian of heading error
ON_FLAG = 0.08      # inimigo a menos disso da bandeira = "em cima" dela
WALL_R = 0.10       # começa a se afastar da parede a 10 cm


# --- telemetria para os logs do torneio ---------------------------------------
# A cada TEL_PERIOD s (relógio do ROS) escreve no log uma linha "TEL {json}" com a
# posição de todos os robôs, bola/bandeiras, estado do jogo e o comando enviado.
# Mudanças de placar/estado do jogo saem na hora como "EVT {json}", e os parâmetros
# do nó saem uma vez como "TELINIT {json}". Os logs aparecem em api/log do site
# do torneio. Isso só gera log: o comportamento do robô continua igual.
TIPO = "classico"
TEL_PERIOD = 2.0  # s entre linhas TEL; 0 desliga (site guarda ~400 linhas)
TEL_IGNORAR = ("field", "ring_radius", "base_x", "base_radius")  # constantes: não vão para o log
TEL_RELOGIO = ("t", "time", "round_time", "held", "message")  # não disparam EVT (vêm junto de placar/tags)


def _tel_r(p, n=3):
    """Arredonda (x, y[, yaw, ...]) para caber no log; None continua None."""
    if p is None:
        return None
    return [round(float(x), n) for x in p]


class Telemetria:
    def __init__(self, period=TEL_PERIOD):
        self.period, self.last, self.ultimo_jogo, self.t0 = period, -1e9, None, None
        self.init, self.eventos = None, True  # eventos=False: este nó não escreve EVT

    @staticmethod
    def _log(tag, campos):
        try:
            rospy.loginfo("%s %s", tag, json.dumps(campos, separators=(",", ":")))
        except Exception as e:  # nunca derrubar o nó por causa do log
            rospy.logwarn_throttle(10.0, "telemetria: %s" % e)

    def inicio(self, eventos=True, **campos):
        self.init, self.eventos = campos, eventos

    def __call__(self, now, game=None, **campos):
        if self.period <= 0:
            return
        if self.t0 is None:
            self.t0 = now
            if self.init is not None:
                self._log("TELINIT", self.init)
        t = round(now - self.t0, 2)  # s desde o início do nó
        game = {k: v for k, v in (game or {}).items() if k not in TEL_IGNORAR}
        chave = json.dumps({k: v for k, v in game.items() if k not in TEL_RELOGIO}, sort_keys=True)
        if self.eventos and self.ultimo_jogo is not None and chave != self.ultimo_jogo:
            self._log("EVT", dict(t=t, jogo=game))
        self.ultimo_jogo = chave
        if now - self.last < self.period:
            return
        self.last = now
        self._log("TEL", dict(t=t, jogo=game, **campos))


class CaptureTheFlag:
    def __init__(self):
        rospy.init_node("ctf")

        # meu robo
        self.robot = rospy.get_param("~robot", 0)
        self.team = rospy.get_param("~team", "blue" if self.robot == 0 else "red")
        self.role = rospy.get_param("~role", "attacker")
        self.ranges = None

        self.opponents = list(rospy.get_param("~opponents", [1 if self.robot == 0 else 0]))
        self.blue = self.team == "blue"
        self.home = (-BASE_X if self.blue else BASE_X, 0.0)  # your base and your flag
        self.me = None  # (x, y, yaw)
        self.opp = {}  # opponent id -> (x, y, yaw)
        self.enemy_flag = None  # (x, y)
        self.own_flag = None  # (x, y): away from self.home means an opponent has it
        self.carrying = False  # True while you hold the enemy flag

        self.pub = rospy.Publisher("/hero_%d/cmd_vel" % self.robot, Twist, queue_size=1)

        enemy = "red" if self.blue else "blue"
        own = "blue" if self.blue else "red"

        rospy.Subscriber("/hero_%d/odom" % self.robot, Odometry, self.on_me)
        for k in self.opponents:
            rospy.Subscriber("/hero_%d/odom" % k, Odometry, self.on_opp, callback_args=k)
        rospy.Subscriber("/flag/" + enemy, PointStamped, self.on_enemy_flag)
        rospy.Subscriber("/flag/" + own, PointStamped, self.on_own_flag)
        rospy.Subscriber("/hero_%d/has_flag" % self.robot, Bool, self.on_has_flag)
        # telemetria: estado do jogo (placar, quem está preso/carregando) e linhas TEL no log
        self.game = {}
        rospy.Subscriber("/game/state", String, self.on_game, queue_size=1)
        self.tel = Telemetria()
        self.tel.inicio(eventos=self.role == "attacker" or len(self.opponents) < 2,
                        jogo="ctf", tipo=TIPO, robot=self.robot, team=self.team, role=self.role,
                        opponents=self.opponents)

    def on_game(self, msg):
        try:
            self.game = json.loads(msg.data)
        except (ValueError, TypeError):
            pass

    def on_laser(self, msg):
        self.ranges = list(msg.ranges)

    def on_me(self, msg):
        # TODO 1 (ROS): self.me = pose_from_odom(msg)
        self.me = pose_from_odom(msg)


    def on_opp(self, msg, k):
        # TODO 1 (ROS): self.opp[k] = pose_from_odom(msg)
        self.opp[k] = pose_from_odom(msg)

    def on_enemy_flag(self, msg):
        # TODO 1 (ROS): self.enemy_flag = (msg.point.x, msg.point.y)
        self.enemy_flag = (msg.point.x, msg.point.y)

    def on_own_flag(self, msg):
        # TODO 1 (ROS): self.own_flag = (msg.point.x, msg.point.y)
        self.own_flag = (msg.point.x, msg.point.y)

    def on_has_flag(self, msg):
        # TODO 1 (ROS): self.carrying = msg.data
        self.carrying = msg.data
    
    def go(self, me, goal, carrying):
        """Controle proporcional até o ponto goal = (gx, gy). Devolve (v, w)."""
        x, y, yaw = me
        gx, gy = goal

        rho = math.hypot(gx - x, gy - y)
        alpha = wrap_angle(math.atan2(gy - y, gx - x) - yaw)

        direction = 1.0

        if abs(alpha) > math.pi /2:
            alpha = wrap_angle(alpha - math.pi)
            direction = -1.0

        w = clamp(K_ALPHA * alpha, -W_MAX, W_MAX)
        v = direction * clamp(K_RHO * rho, 0.0, V_MAX) * clamp(math.cos(alpha), 0.0, 1.0)

        return v, w

    def avoid(self, me, goal, opponents):
        """Devolve um ponto intermediário que leva ao goal desviando dos inimigos perigosos."""
        x, y, _ = me
        gx, gy = goal

        # atração
        d = math.hypot(gx - x, gy - y)
        if d < 1e-6:
            return goal
        fx, fy = (gx - x) / d, (gy - y) / d
        fx0, fy0 = fx, fy 

        in_enemy_half = (x > 0) if self.blue else (x < 0)

        if in_enemy_half:
            for ox, oy, _ in opponents:
                if ((ox > 0) if self.blue else (ox < 0)):
                    dist = math.hypot(x - ox, y - oy)
                    if 1e-6 < dist < AVOID_R:
                        # campo potencial de repulsao do enimigo
                        force = K_REP * (AVOID_R / dist - 1.0)
                        rx, ry = (x - ox) / dist, (y - oy) / dist 
                        tx, ty = -ry, rx
                        dot = tx * fx0 + ty * fy0
                        if abs(dot) < 0.2:
                            side = 1.0 if (y - oy) >= 0 else -1.0
                            if ty * side < 0:
                                tx, ty = -tx, -ty
                        elif dot < 0:
                            tx, ty = -tx, -ty
                        fx += force * (rx + K_TAN * tx)
                        fy += force * (ry + K_TAN * ty)

        # ponto intermediário
        n = math.hypot(fx, fy)
        step = min(0.15, d)

        x = x + step * fx / n
        y = y + step * fy / n

        return (x, y)



    def compute_command(self, me, opponents, enemy_flag, own_flag, carrying):
        """TODO 2: choose (linear_x, angular_z).

        me: (x, y, yaw). opponents: list of (x, y, yaw). flags: (x, y). carrying: you hold the enemy flag.
        self.role tells whether you attack (go for the flag, run home, avoid being tagged in the
        opponents' half) or defend (stay between your flag and the opponents, chase intruders).
        """
        x, y, yaw = me

        if self.role == "attacker":
            
            if carrying:
                target = self.home                      
                goal = self.avoid(me, target, opponents)

            else:
                # ataca se o inimigo esta em cima da bandeira
                fx, fy = enemy_flag
                guard = min(math.hypot(ox - fx, oy - fy) for ox, oy, _ in opponents)
                if guard < ON_FLAG:
                    goal = enemy_flag          
                else:
                    goal = self.avoid(me, enemy_flag, opponents)
            
        else:
            
            # ponto entre bandeira e inimigo mais proximo
            ox, oy, _ = min(opponents, key=lambda o: math.hypot(o[0]-own_flag[0], o[1]-own_flag[1]))
            
            # se pegaram bandeira vai atras
            if math.hypot(own_flag[0] - self.home[0], own_flag[1] - self.home[1]) > 0.05:
                goal = (ox, oy)            
            else:
                # fica entre bandeira e inimigo, usando 10 cm de distancia da bandeira
                ang = math.atan2(oy - own_flag[1], ox - own_flag[0])
                goal = (own_flag[0] + 0.10*math.cos(ang), own_flag[1] + 0.10*math.sin(ang))
        
        v, w = self.go(me, goal, carrying) 
        
        return v, w

    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if (self.me is None or len(self.opp) < len(self.opponents)
                    or self.enemy_flag is None or self.own_flag is None):
                rospy.logwarn_throttle(2.0, "waiting for odometry and flags (TODO 1)")
            else:
                v, w = self.compute_command(self.me, list(self.opp.values()), self.enemy_flag,
                                            self.own_flag, self.carrying)
                cmd = Twist()
                cmd.linear.x, cmd.angular.z = v, w
                self.pub.publish(cmd)
                self.tel(rospy.get_time(), self.game, me=_tel_r(self.me), carrying=bool(self.carrying),
                         opp={k: _tel_r(p) for k, p in sorted(self.opp.items())},
                         enemy_flag=_tel_r(self.enemy_flag), own_flag=_tel_r(self.own_flag),
                         cmd=_tel_r((v, w)))
            rate.sleep()


if __name__ == "__main__":
    CaptureTheFlag().run()
