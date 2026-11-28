"""Avaliação: ctf_classico_anterior.py com o /game/state repassado ao on_game()."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _game_wrapper import wrap
_mod, CaptureTheFlag = wrap("../ctf/anteriores/ctf_classico_anterior.py")
CONTROL_HZ = 20
