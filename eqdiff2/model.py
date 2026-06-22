"""
Modelo de rede trofica Puma - Guanaco - Ovelha (Patagonia)
===========================================================

Sistema de equacoes diferenciais ordinarias (EDO) acopladas, uma extensao
das equacoes de Lotka-Volterra para UM predador e DUAS presas que competem
entre si.

Inspirado em: "Coexistence of pumas, guanacos and sheep in Patagonia"
(arXiv:2412.02936v1).

Variaveis de estado
-------------------
    P  -> populacao de PUMAS    (predador)
    G  -> populacao de GUANACOS (presa nativa)
    O  -> populacao de OVELHAS  (presa introduzida)

Equacoes
--------
    dP/dt = -m_P * P + e_G * P * G + e_O * P * O
    dG/dt =  r_G * G * (1 - G/K_G) - a_G * P * G - c_GO * G * O
    dO/dt =  r_O * O * (1 - O/K_O) - a_O * P * O - c_OG * O * G

Obs.: a equacao das ovelhas no enunciado original tinha um erro de digitacao
      "(1 - K_O/K_O)"; o termo logistico correto e "(1 - O/K_O)".

A integracao numerica e feita pelo METODO DE EULER (explicito), o esquema de
passo unico mais simples:

    y_{n+1} = y_n + dt * f(y_n)
"""

from dataclasses import dataclass, asdict, field
import numpy as np


# ---------------------------------------------------------------------------
# Parametros do modelo
# ---------------------------------------------------------------------------
@dataclass
class Params:
    # --- Pumas (predador) ---
    m_P: float = 0.30   # taxa de mortalidade natural dos pumas
    e_G: float = 0.0008  # eficiencia de conversao ao cacar guanacos
    e_O: float = 0.0006  # eficiencia de conversao ao cacar ovelhas

    # --- Guanacos (presa nativa) ---
    r_G: float = 0.80   # taxa de crescimento intrinseco
    K_G: float = 500.0  # capacidade de suporte
    a_G: float = 0.0030  # taxa de ataque do puma ao guanaco
    c_GO: float = 0.0007  # competicao: efeito das ovelhas sobre os guanacos

    # --- Ovelhas (presa introduzida) ---
    r_O: float = 0.70   # taxa de crescimento intrinseco
    K_O: float = 500.0  # capacidade de suporte
    a_O: float = 0.0040  # taxa de ataque do puma a ovelha (mais vulneravel)
    c_OG: float = 0.0004  # competicao: efeito dos guanacos sobre as ovelhas

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Campo vetorial  f(estado) = (dP/dt, dG/dt, dO/dt)
# ---------------------------------------------------------------------------
def derivatives(P, G, O, p: Params):
    """Retorna (dP/dt, dG/dt, dO/dt) para o estado (P, G, O)."""
    dP = -p.m_P * P + p.e_G * P * G + p.e_O * P * O
    dG = p.r_G * G * (1.0 - G / p.K_G) - p.a_G * P * G - p.c_GO * G * O
    dO = p.r_O * O * (1.0 - O / p.K_O) - p.a_O * P * O - p.c_OG * O * G
    return dP, dG, dO


def euler_step(state, p: Params, dt: float):
    """Um unico passo do metodo de Euler explicito.

    state : tupla (P, G, O)
    p     : Params
    dt    : passo de tempo
    """
    P, G, O = state
    dP, dG, dO = derivatives(P, G, O, p)
    P_new = P + dt * dP
    G_new = G + dt * dG
    O_new = O + dt * dO
    # populacoes nao podem ser negativas (Euler pode "ultrapassar" o zero)
    return (max(P_new, 0.0), max(G_new, 0.0), max(O_new, 0.0))


def simulate(p: Params,
             P0: float = 60.0, G0: float = 300.0, O0: float = 200.0,
             dt: float = 0.02, t_final: float = 300.0):
    """Integra o sistema com o metodo de Euler.

    Retorna (t, P, G, O) como arrays numpy.
    """
    n_steps = int(round(t_final / dt))
    t = np.empty(n_steps + 1)
    P = np.empty(n_steps + 1)
    G = np.empty(n_steps + 1)
    O = np.empty(n_steps + 1)

    t[0], P[0], G[0], O[0] = 0.0, P0, G0, O0
    state = (P0, G0, O0)
    for n in range(n_steps):
        state = euler_step(state, p, dt)
        P[n + 1], G[n + 1], O[n + 1] = state
        t[n + 1] = t[n] + dt
    return t, P, G, O


if __name__ == "__main__":
    # Teste rapido / diagnostico de estabilidade dos parametros padrao
    p = Params()
    t, P, G, O = simulate(p)
    print(f"Passos: {len(t)}  |  t_final = {t[-1]:.1f}")
    print(f"Estado final  -> P={P[-1]:8.2f}  G={G[-1]:8.2f}  O={O[-1]:8.2f}")
    print(f"Maximos       -> P={P.max():8.2f}  G={G.max():8.2f}  O={O.max():8.2f}")
    print(f"Minimos       -> P={P.min():8.2f}  G={G.min():8.2f}  O={O.min():8.2f}")
    finite = np.isfinite(P).all() and np.isfinite(G).all() and np.isfinite(O).all()
    print(f"Bem comportado (sem inf/nan): {finite}")
