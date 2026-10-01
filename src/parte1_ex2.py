"""
Parte 1, exercicio 2: rede de resistores com duas fontes (sistema 4x4).

1) conferimos a dominancia diagonal (criterio das linhas, slide 30) e o
   criterio de Sassenfeld (slide 31);
2) resolvemos por Gauss-Jacobi (slide 23) e Gauss-Seidel (slide 27) com o
   criterio de parada relativo do slide 24;
3) comparamos com um metodo direto: Gauss com pivotamento (slides 14 e 18).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metodos import (criterio_linhas, gauss_pivoteamento, gauss_seidel, jacobi,
                     matrizes_iteracao, norma_inf, num, pasta_resultados,
                     raio_espectral, salvar_csv, salvar_tabela_tex,
                     salvar_valores, sassenfeld)

R1 = R2 = R3 = 100.0
R4 = R5 = 150.0
R6 = R7 = R8 = 200.0
V1, V2 = 10.0, 12.0

A = np.array([[R1 + R2 + R4, -R2, 0.0, -R4],
              [-R2, R2 + R3 + R5, -R5, 0.0],
              [0.0, -R5, R5 + R7 + R8, -R7],
              [-R4, 0.0, -R7, R4 + R6 + R7]])
b = np.array([-V1, V2, 0.0, 0.0])

TOL = 1e-6        # tolerancia do erro relativo (slide 24)
KMAX = 200
x0 = np.zeros(4)

AZUL, LARANJA = "#2a78d6", "#eb6834"       # paleta de referencia (slots 1 e 2)
TINTA, TINTA2 = "#0b0b0b", "#52514e"


def tabela_iteracoes(hist, x_ref):
    linhas, csv_l = [], []
    for k, x, d in hist:
        e = norma_inf(x - x_ref)
        linhas.append([str(k)] + [num(v, casas=7) for v in x] +
                      [num(d, sci=2), num(e, sci=2)])
        csv_l.append([k, *x, d, e])
    return linhas, csv_l


def main():
    print("Parte 1 - Ex. 2 (rede de resistores)")
    print("A =\n", A)

    # ---- dominancia diagonal / criterios de convergencia
    alfas, alfa = criterio_linhas(A)
    betas, beta = sassenfeld(A)
    diag = np.abs(np.diag(A))
    fora = np.abs(A).sum(axis=1) - diag
    print("  |a_ii|           :", diag)
    print("  sum_{j!=i}|a_ij| :", fora)
    print("  alfas =", alfas, " alfa =", alfa)
    print("  betas =", betas, " beta =", beta)
    CJ, CGS = matrizes_iteracao(A)
    rhoJ, rhoGS = raio_espectral(CJ), raio_espectral(CGS)
    print("  rho(C_J) = %.4f   rho(C_GS) = %.4f" % (rhoJ, rhoGS))

    # ---- metodo direto
    x_dir, trocas = gauss_pivoteamento(A, b)
    x_np = np.linalg.solve(A, b)
    print("  Gauss c/ pivotamento:", x_dir, " trocas:", trocas)
    print("  ||x_gauss - x_numpy||_inf = %.2e" % norma_inf(x_dir - x_np))

    # ---- iterativos
    xJ, hJ, pJ = jacobi(A, b, x0, TOL, KMAX)
    xS, hS, pS = gauss_seidel(A, b, x0, TOL, KMAX)
    kJ, kS = hJ[-1][0], hS[-1][0]
    resJ = norma_inf(b - A @ xJ)
    resS = norma_inf(b - A @ xS)
    eJ = norma_inf(xJ - x_dir)
    eS = norma_inf(xS - x_dir)
    print("  Jacobi: %d iteracoes (%s), residuo %.2e, erro %.2e" % (kJ, pJ, resJ, eJ))
    print("  Seidel: %d iteracoes (%s), residuo %.2e, erro %.2e" % (kS, pS, resS, eS))

    # ---- tabelas
    linhas = []
    for i in range(4):
        linhas.append([str(i + 1), num(diag[i], casas=0), num(fora[i], casas=0),
                       num(alfas[i], casas=4), num(betas[i], casas=4)])
    salvar_tabela_tex("ex2_criterios.tex", "crrrr",
                      [r"Linha $i$", r"$|a_{ii}|$", r"$\sum_{j\neq i}|a_{ij}|$",
                       r"$\alpha_i$", r"$\beta_i$"], linhas)

    cab = [r"$k$", r"$i_1^{(k)}$", r"$i_2^{(k)}$", r"$i_3^{(k)}$", r"$i_4^{(k)}$",
           r"$d^{(k)}$", r"$\|x^{(k)}-x\|_\infty$"]
    for nome, hist in (("jacobi", hJ), ("seidel", hS)):
        lt, lc = tabela_iteracoes(hist, x_dir)
        salvar_tabela_tex("ex2_%s.tex" % nome, "rrrrrrr", cab, lt)
        salvar_csv("ex2_%s.csv" % nome,
                   ["k", "i1", "i2", "i3", "i4", "d_rel", "erro_inf"], lc)

    linhas = [["Gauss c/ pivotamento"] + [num(v, casas=7) for v in x_dir] + ["", ""],
              ["Gauss-Jacobi"] + [num(v, casas=7) for v in xJ] + [str(kJ), num(resJ, sci=1)],
              ["Gauss-Seidel"] + [num(v, casas=7) for v in xS] + [str(kS), num(resS, sci=1)]]
    salvar_tabela_tex("ex2_comparacao.tex", "lrrrrrr",
                      ["M\\'etodo", r"$i_1$", r"$i_2$", r"$i_3$", r"$i_4$",
                       "Iter.", r"$\|b-Ax\|_\infty$"], linhas)
    salvar_csv("ex2_comparacao.csv", ["metodo", "i1", "i2", "i3", "i4", "iter", "residuo"],
               [["gauss", *x_dir, "", norma_inf(b - A @ x_dir)],
                ["jacobi", *xJ, kJ, resJ], ["seidel", *xS, kS, resS]])

    # ---- figura: erro por iteracao (como o slide 29)
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for hist, cor, mk, rot in ((hJ, AZUL, "o", "Gauss-Jacobi"),
                               (hS, LARANJA, "s", "Gauss-Seidel")):
        ks = [h[0] for h in hist]
        es = [norma_inf(h[1] - x_dir) for h in hist]
        ax.semilogy(ks, es, color=cor, lw=2, marker=mk, ms=5, label=rot)
        ax.annotate(rot, (ks[-1], es[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=TINTA)
    ax.set_xlabel("iteração $k$", color=TINTA2)
    ax.set_ylabel(r"$\|x^{(k)} - x\|_\infty$", color=TINTA2)
    ax.grid(True, which="major", color="#e4e3df", lw=0.6)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#b5b4ae")
    ax.tick_params(colors=TINTA2)
    ax.legend(frameon=False, loc="upper right")
    ax.set_xlim(0, max(kJ, kS) + 4)
    fig.tight_layout()
    fig.savefig(pasta_resultados() / "ex2_convergencia.pdf")
    plt.close(fig)

    mat = lambda M: r" \\ ".join(" & ".join(num(v, casas=0) for v in lin) for lin in M)
    salvar_valores("ex2_valores.tex", {
        "exDoisA": mat(A),
        "exDoisb": r" \\ ".join(num(v, casas=0) for v in b),
        "exDoisKmax": str(KMAX),
        "exDoisTol": num(TOL, sci=0),
        "exDoisAlfa": num(alfa, casas=4),
        "exDoisBeta": num(beta, casas=4),
        "exDoisRhoJ": num(rhoJ, casas=4),
        "exDoisRhoGS": num(rhoGS, casas=4),
        "exDoisItJ": str(kJ),
        "exDoisItS": str(kS),
        "exDoisErroJ": num(eJ, sci=1),
        "exDoisErroS": num(eS, sci=1),
        "exDoisTrocas": str(trocas),
        "exDoisDifNp": num(norma_inf(x_dir - x_np), sci=1),
        "exDoisResDir": num(norma_inf(b - A @ x_dir), sci=1),
        "exDoisIi": num(x_dir[0], casas=7),
        "exDoisIii": num(x_dir[1], casas=7),
        "exDoisIiii": num(x_dir[2], casas=7),
        "exDoisIiv": num(x_dir[3], casas=7),
        "exDoisIimA": num(1000 * x_dir[0], casas=2),
        "exDoisIiimA": num(1000 * x_dir[1], casas=2),
        "exDoisIiiimA": num(1000 * x_dir[2], casas=2),
        "exDoisIivmA": num(1000 * x_dir[3], casas=2),
    })
    print("arquivos gravados em resultados/")


if __name__ == "__main__":
    main()
