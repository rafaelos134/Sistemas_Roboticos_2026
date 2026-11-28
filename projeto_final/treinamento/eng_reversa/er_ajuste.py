"""Ajuste dos parâmetros de uma máquina de estados paramétrica aos comandos deduzidos das gravações.

Erro = média de |v_pred - v_real|/V_MAX e |w_pred - w_real|/W_MAX (os dois cortados em [-1, 1]), o
mesmo do nyx_validar.py e do train_ctf_bc.py. Validação: 25 % das partidas separadas (por id).

Busca: aleatória (amostras do espaço) e depois descida por coordenada (grade em cada parâmetro), 3 voltas.
"""
from __future__ import annotations

import numpy as np

import er_dados as E


def erro(v, w, vr, wr, por_comp=False):
    a = np.abs(np.clip(v / E.V_MAX, -1, 1) - np.clip(vr / E.V_MAX, -1, 1))
    b = np.abs(np.clip(w / E.W_MAX, -1, 1) - np.clip(wr / E.W_MAX, -1, 1))
    if por_comp:
        return float(a.mean()), float(b.mean())
    return float((a.mean() + b.mean()) / 2)


def erro_constante(vr, wr):
    """Erro de prever sempre o comando médio (referência mínima)."""
    return erro(np.full_like(vr, vr.mean()), np.full_like(wr, wr.mean()), vr, wr)


def separar(mid, frac=0.25, seed=0):
    ids = np.unique(mid)
    rng = np.random.default_rng(seed)
    val = set(rng.choice(ids, max(1, int(round(frac * len(ids)))), replace=False).tolist())
    m = np.array([x in val for x in mid])
    return ~m, m


def _sub(F, m):
    return {k: (x[m] if isinstance(x, np.ndarray) and x.shape[:1] == m.shape else x) for k, x in F.items()}


def ajustar(fam, F, vr, wr, mid, n_aleat=400, voltas=3, seed=0, fixos=None, log=print):
    """fam: módulo com ESPACO {nome: (tipo, valores)} e politica(F, P) -> (v, w)."""
    tr, va = separar(mid, seed=seed)
    Ft, Fv = _sub(F, tr), _sub(F, va)
    rng = np.random.default_rng(seed)
    esp = dict(fam.ESPACO)
    fixos = fixos or {}

    def f(P, FF=Ft, m=tr):
        v, w = fam.politica(FF, P)
        return erro(v, w, vr[m], wr[m])

    def sortear():
        P = {}
        for k, vals in esp.items():
            P[k] = vals[rng.integers(len(vals))]
        P.update(fixos)
        return P

    melhor = dict(fam.PADRAO)
    melhor.update(fixos)
    eb = f(melhor)
    for _ in range(n_aleat):
        P = sortear()
        e = f(P)
        if e < eb:
            melhor, eb = P, e
    for volta in range(voltas):
        mudou = False
        for k, vals in esp.items():
            if k in fixos:
                continue
            for x in vals:
                P = dict(melhor)
                P[k] = x
                e = f(P)
                if e < eb - 1e-6:
                    melhor, eb, mudou = P, e, True
        if not mudou:
            break
    v, w = fam.politica(Fv, melhor)
    res = {"treino": eb, "validacao": erro(v, w, vr[va], wr[va]),
           "validacao_v_w": erro(v, w, vr[va], wr[va], por_comp=True),
           "constante_validacao": erro_constante(vr[va], wr[va]), "n": int(len(vr)),
           "partidas": int(len(np.unique(mid)))}
    return melhor, res
