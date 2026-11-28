"""Usado só na avaliação offline (rl/eval_ctf.py): o avaliador preenche node.game e node.now
mas não chama on_game(). Este wrapper repassa esse estado ao on_game() do nó, acrescentando o
campo "t" (tempo do simulador), como faz a ponte ROS do kit em /game/state. A estratégia não muda."""
import importlib.util
import json
import os
import types


def wrap(filename):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location("inner_" + os.path.basename(filename)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class CaptureTheFlag(mod.CaptureTheFlag):
        def compute_command(self, *args):
            g = dict(getattr(self, "game", None) or {})
            if g and hasattr(self, "on_game"):
                g["t"] = self.now
                self.on_game(types.SimpleNamespace(data=json.dumps(g)))
            return super().compute_command(*args)

    return mod, CaptureTheFlag
