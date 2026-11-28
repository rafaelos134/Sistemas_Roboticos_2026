"""Avaliação: ctf_classico.py (Guilherme) com o /game/state repassado ao on_game()."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _game_wrapper import wrap
_mod, CaptureTheFlag = wrap("../ctf/ctf_classico.py")
CONTROL_HZ = 20
