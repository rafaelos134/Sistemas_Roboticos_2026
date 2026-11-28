"""Avaliação: soccer_classico.py no rl/eval_soccer.py. O avaliador preenche node.ball_v,
mas o nó clássico lê a velocidade da bola em node.ball_vel (preenchida por on_ball() no ROS).
Este wrapper só copia um campo para o outro; a estratégia não muda."""
import importlib.util
import os

_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "soccer", "soccer_classico.py")
_spec = importlib.util.spec_from_file_location("inner_soccer_classico", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


class Soccer(_mod.Soccer):
    def compute_commands(self, poses, ball):
        self.ball_vel = tuple(getattr(self, "ball_v", (0.0, 0.0)))
        return super().compute_commands(poses, ball)
