#!/usr/bin/env bash
# Sumô RL, versão final (v6 = sumo_ppo_v6_fus): sensores com ruído + EKF dos dois robôs, que gerou o
# projeto_final/rl/sumo.py (iteração 800). Parte do modelo da v5 e do grupo de cópias congeladas da v5,
# incluídos em modelos/sumo_ppo_v6_fus/init_v5/. Requer GPU (Genesis + PyTorch) e o kit em ../kit (ou PSR_KIT).
set -e; cd "$(dirname "$0")"; PY=${PY:-python3}
I=modelos/sumo_ppo_v6_fus/init_v5
$PY train_sumo.py --phase 5 --fusion --iters 1500 --name sumo_ppo_v6_fus --init $I/model.pt --pool-from $I/pool
mkdir -p exports
# a versão entregue é a iteração 800: depois dela o desvio-padrão da ação angular satura em 1
$PY export_sumo.py runs/sumo_ppo_v6_fus/pool/it0800.pt exports/sumo.py --fusion
