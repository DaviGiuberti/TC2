"""
Parte 1, exercicio 3 (UFRGS E 5.1.3): intersecao da parabola y = x^2 + 1
com a elipse x^2 + y^2/4 = 1, pelo metodo de Newton.

    F(x, y) = ( x^2 + 1 - y ,  x^2 + y^2/4 - 1 )
    J_F     = [[2x, -1], [2x, y/2]],   det J_F = x (y + 2)

Os chutes iniciais saem do esboco (figura ex3_esboco.pdf).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metodos import (erros_e_ordem, newton, norma_inf, num, pasta_resultados,
                     salvar_csv, salvar_tabela_tex, salvar_valores,
                     solucao_referencia)

TOL = 1e-10       # ||s^(k)||_inf
TOL_F = 1e-12     # ||F(x^(k))||_inf
KMAX = 50

# chutes lidos do esboco
CHUTES = {"Q1": np.array([0.7, 1.5]),     # 1o quadrante
          "Q2": np.array([-0.7, 1.5])}    # 2o quadrante

AZUL, LARANJA = "#2a78d6", "#eb6834"
TINTA, TINTA2 = "#0b0b0b", "#52514e"


def F(v):
    x, y = v
    return np.array([x**2 + 1 - y,
                     x**2 + y**2 / 4 - 1])


def JF(v):
    x, y = v
    return np.array([[2 * x, -1.0],
                     [2 * x, y / 2]])


def esboco(raizes):
    t = np.linspace(0, 2 * np.pi, 400)
    xs = np.linspace(-1.6, 1.6, 400)
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.axhline(0, color="#b5b4ae", lw=0.8)
    ax.axvline(0, color="#b5b4ae", lw=0.8)
    ax.plot(xs, xs**2 + 1, color=AZUL, lw=2, label=r"parábola $y = x^2 + 1$")
    ax.plot(np.cos(t), 2 * np.sin(t), color=LARANJA, lw=2, ls="--",
            label=r"elipse $x^2 + y^2/4 = 1$")
    for r in raizes:
        ax.plot(*r, marker="o", ms=11, mfc="none", mec=TINTA, mew=1.6, zorder=5)
        lado = 1 if r[0] > 0 else -1
        ax.annotate(("(%.4f; %.4f)" % (r[0], r[1])).replace(".", ","),
                    r, xytext=(lado * 12, 14), textcoords="offset points",
                    ha="left" if lado > 0 else "right", fontsize=9, color=TINTA)
    for nome, x0 in CHUTES.items():
        ax.plot(*x0, marker="x", ms=7, mew=1.8, color=TINTA2, zorder=6)
    ax.plot([], [], marker="x", ls="", ms=7, mew=1.8, color=TINTA2, label="chute inicial")
    ax.plot([], [], marker="o", ls="", ms=11, mfc="none", mec=TINTA, mew=1.6,
            label="raiz (Newton)")
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-2.3, 3.2)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$", color=TINTA2)
    ax.set_ylabel("$y$", color=TINTA2)
    ax.grid(True, color="#e4e3df", lw=0.6)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(colors=TINTA2)
    ax.legend(frameon=False, loc="lower center", fontsize=8.5, ncol=2,
              bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(pasta_resultados() / "ex3_esboco.pdf")
    plt.close(fig)


def main():
    print("Parte 1 - Ex. 3 (parabola x elipse)")
    # solucao exata para conferir: x^2 = y - 1 na elipse -> y^2 + 4y - 8 = 0
    y_ex = -2 + 2 * np.sqrt(3)
    x_ex = np.sqrt(y_ex - 1)
    print("  solucao exata: x = +-%.15f, y = %.15f" % (x_ex, y_ex))

    valores = {"exTresTol": num(TOL, sci=0), "exTresTolF": num(TOL_F, sci=0),
               "exTresxEx": num(x_ex, casas=12), "exTresyEx": num(y_ex, casas=12)}
    raizes = []
    for nome, x0 in CHUTES.items():
        x, hist, parada = newton(F, JF, x0, TOL, KMAX, TOL_F)
        xstar = solucao_referencia(F, JF, x)
        e, expo, p = erros_e_ordem(hist, xstar)
        raizes.append(x)
        print("  chute %s = %s -> x = %s, %d iteracoes, parada: %s"
              % (nome, x0, x, hist[-1]["k"], parada))
        print("    |x* - exata|_inf = %.2e" % norma_inf(xstar - np.array([np.sign(x0[0]) * x_ex, y_ex])))
        linhas, lcsv = [], []
        for h, ek, xk, pk in zip(hist, e, expo, p):
            linhas.append([str(h["k"]), num(h["x"][0], casas=12), num(h["x"][1], casas=12),
                           num(h["normaF"], sci=1), num(h["normaS"], sci=1),
                           num(ek, sci=1), "" if xk is None else "$%d$" % xk,
                           num(pk, casas=2)])
            lcsv.append([h["k"], *h["x"], h["normaF"], h["normaS"], ek, pk])
            print("    k=%d  x=(%.12f, %.12f)  |F|=%.1e  e=%.1e  p=%s"
                  % (h["k"], *h["x"], h["normaF"], ek, "%.2f" % pk if pk == pk else "-"))
        salvar_tabela_tex("ex3_%s.tex" % nome, "rrrrrrrr",
                          [r"$k$", r"$x^{(k)}$", r"$y^{(k)}$",
                           r"$\|F(x^{(k)})\|_\infty$", r"$\|s^{(k)}\|_\infty$",
                           r"$\|x^{(k)}-x^*\|_\infty$", "exp.", r"$p_k$"], linhas)
        salvar_csv("ex3_%s.csv" % nome, ["k", "x", "y", "normaF", "normaS", "erro", "p"], lcsv)
        s = "exTres" + nome.replace("Q1", "Um").replace("Q2", "Dois")
        valores[s + "x"] = num(x[0], casas=10)
        valores[s + "y"] = num(x[1], casas=10)
        valores[s + "It"] = str(hist[-1]["k"])
        valores[s + "Parada"] = {"passo": r"$\|s^{(k)}\|_\infty$",
                                 "residuo": r"$\|F(x^{(k)})\|_\infty$",
                                 "kmax": "n\\'umero m\\'aximo de itera\\c{c}\\~oes"}[parada]
        valores[s + "ErroExato"] = num(norma_inf(x - np.array([np.sign(x0[0]) * x_ex, y_ex])), sci=1)
        valores[s + "P"] = num([v for v in p if v == v][-1], casas=2)

    # o que acontece com um chute em cima do eixo y (x = 0): J singular
    try:
        newton(F, JF, np.array([0.0, 1.5]), TOL, KMAX, TOL_F)
        msg = "(convergiu)"
    except np.linalg.LinAlgError as err:
        msg = str(err)
    print("  chute (0; 1,5): numpy.linalg.LinAlgError:", msg)
    valores["exTresErroSing"] = r"\texttt{%s}" % msg

    esboco(raizes)
    salvar_valores("ex3_valores.tex", valores)
    print("arquivos gravados em resultados/")


if __name__ == "__main__":
    main()
