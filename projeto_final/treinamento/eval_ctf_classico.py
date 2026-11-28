"""Avaliação: ../classico/ctf.py (versão clássica) com o /game/state repassado ao on_game().
    python3 eval_ctf.py --blue ../rl/ctf.py --red eval_ctf_classico.py --minutes 5 --matches 3"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _game_wrapper import wrap
_mod, CaptureTheFlag = wrap("../classico/ctf.py")
CONTROL_HZ = 20
