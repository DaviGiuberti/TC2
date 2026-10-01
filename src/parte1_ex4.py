"""
Parte 1, exercicio 4 (UFRGS E 5.1.5): solucao perto da origem de

    6x - 2y + e^z = 2
    sen x - y + z = 0
    sen x + 2y + 3z = 1

com erro menor que 1e-5 em cada incognita.

Criterio usado: paramos quando ||s^(k)||_inf < 1e-5 (e checamos o residuo).
Perto da raiz, com convergencia quadratica, ||s^(k)|| = ||x^(k+1) - x^(k)||
e praticamente igual ao erro de x^(k), e o erro de x^(k+1) (que e o valor
devolvido) e da ordem de ||s^(k)||^2. Depois conferimos o erro de verdade
contra uma solucao de referencia no limite da precisao de maquina.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metodos import (erros_e_ordem, newton, norma_inf, num, salvar_csv,
                     salvar_tabela_tex, salvar_valores, solucao_referencia)

TOL = 1e-5        # exigencia do enunciado, aplicada em ||s^(k)||_inf
TOL_F = 1e-10     # residuo (slide 25)
KMAX = 50
x0 = np.zeros(3)  # "solucao proxima da origem"


def F(v):
    x, y, z = v
    return np.array([6 * x - 2 * y + np.exp(z) - 2,
                     np.sin(x) - y + z,
                     np.sin(x) + 2 * y + 3 * z - 1])


def JF(v):
    x, y, z = v
    return np.array([[6.0, -2.0, np.exp(z)],
                     [np.cos(x), -1.0, 1.0],
                     [np.cos(x), 2.0, 3.0]])


def main():
    print("Parte 1 - Ex. 4")
    x, hist, parada = newton(F, JF, x0, TOL, KMAX, TOL_F)
    xstar = solucao_referencia(F, JF, x)
    e, expo, p = erros_e_ordem(hist, xstar)
    erro_final = np.abs(x - xstar)
    print("  x = %s, %d iteracoes, parada: %s" % (x, hist[-1]["k"], parada))
    print("  x* (referencia) = %s" % xstar)
    print("  |x - x*| por componente:", erro_final)
    print("  ||F(x)||_inf = %.2e" % norma_inf(F(x)))

    linhas, lcsv = [], []
    for h, ek, xk, pk in zip(hist, e, expo, p):
        linhas.append([str(h["k"])] + [num(v, casas=10) for v in h["x"]] +
                      [num(h["normaF"], sci=1), num(h["normaS"], sci=1),
                       num(ek, sci=1), "" if xk is None else "$%d$" % xk])
        lcsv.append([h["k"], *h["x"], h["normaF"], h["normaS"], ek, pk])
        print("    k=%d  x=%s  |F|=%.1e  |s|=%.1e  e=%.1e"
              % (h["k"], h["x"], h["normaF"], h["normaS"], ek))
    salvar_tabela_tex("ex4_iteracoes.tex", "rrrrrrrr",
                      [r"$k$", r"$x^{(k)}$", r"$y^{(k)}$", r"$z^{(k)}$",
                       r"$\|F(x^{(k)})\|_\infty$", r"$\|s^{(k)}\|_\infty$",
                       r"$\|x^{(k)}-x^*\|_\infty$", "exp."], linhas)
    salvar_csv("ex4_iteracoes.csv", ["k", "x", "y", "z", "normaF", "normaS", "erro", "p"], lcsv)

    # a mesma conta parando so quando o passo fica no nivel da maquina, para
    # comparar com o criterio de 1e-5
    ult_s = hist[-2]["normaS"]
    valores = {
        "exQuatroTol": num(TOL, sci=0),
        "exQuatroTolF": num(TOL_F, sci=0),
        "exQuatroIt": str(hist[-1]["k"]),
        "exQuatrox": num(x[0], casas=6), "exQuatroy": num(x[1], casas=6),
        "exQuatroz": num(x[2], casas=6),
        "exQuatroxLongo": num(xstar[0], casas=12), "exQuatroyLongo": num(xstar[1], casas=12),
        "exQuatrozLongo": num(xstar[2], casas=12),
        "exQuatroUltS": num(ult_s, sci=1),
        "exQuatroUltSQuad": num(ult_s**2, sci=1),
        "exQuatroResFinal": num(norma_inf(F(x)), sci=1),
        "exQuatroErrox": num(erro_final[0], sci=1),
        "exQuatroErroy": num(erro_final[1], sci=1),
        "exQuatroErroz": num(erro_final[2], sci=1),
        "exQuatroErroMax": num(erro_final.max(), sci=1),
        "exQuatroParada": {"passo": r"$\|s^{(k)}\|_\infty$", "residuo": r"$\|F(x^{(k)})\|_\infty$",
                           "kmax": "n\\'umero m\\'aximo de itera\\c{c}\\~oes"}[parada],
        "exQuatroPenultS": num(hist[-3]["normaS"], sci=1),
        "exQuatroErroPenult": num(e[-2], sci=1),
    }
    salvar_valores("ex4_valores.tex", valores)
    print("arquivos gravados em resultados/")


if __name__ == "__main__":
    main()
