"""
Parte 1, exercicio 1: custo anual de um aluno em cada college.

Interpretacao (ver relatorio): as COLUNAS da tabela somam 100%, entao a
coluna j diz de onde vem os alunos matriculados nos cursos do college j.
Os 4000, 1000 e 2000 sao as matriculas nos cursos de cada college (a frase
"70% from Sciences, 20% from Engineering, 10% from CS" e a primeira coluna).
Se x_i e o custo anual de um aluno do college i, o orcamento do college j
paga todos os alunos que ele atende:

    sum_i  P[i, j] * N[j] * x_i = B[j],   j = 1, 2, 3

Resolvemos por eliminacao de Gauss com pivotamento parcial (slides 14 e 18)
e conferimos com np.linalg.solve.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metodos import (gauss_pivoteamento, norma_inf, num, salvar_csv,
                     salvar_tabela_tex, salvar_valores)

colleges = ["Sciences", "Engineering", "Computer Science"]
B = np.array([16e6, 5e6, 8e6])             # orcamentos (US$)
N = np.array([4000.0, 1000.0, 2000.0])     # matriculas nos cursos de cada college
# P[i, j] = fracao dos alunos dos cursos do college j que vem do college i
P = np.array([[0.70, 0.10, 0.15],
              [0.20, 0.90, 0.10],
              [0.10, 0.00, 0.75]])


def main():
    print("Parte 1 - Ex. 1 (custo por aluno)")
    print("soma das colunas de P:", P.sum(axis=0))

    # linha j do sistema: alunos de cada origem nos cursos do college j
    A = (P * N).T
    print("A =\n", A)
    print("b =", B)

    x, trocas = gauss_pivoteamento(A, B)
    x_np = np.linalg.solve(A, B)
    res = norma_inf(B - A @ x)
    dif = norma_inf(x - x_np)
    x_cent = np.round(x, 2)

    # conferencia de consistencia: alunos de cada college (somando todos os
    # cursos) vezes o custo de cada um tem que dar o orcamento total
    alunos = (P * N).sum(axis=1)
    total = float(alunos @ x)

    for c, xi, xc in zip(colleges, x, x_cent):
        print("  %-17s x = %.6f  ->  US$ %.2f" % (c, xi, xc))
    print("  trocas de linha no pivotamento:", trocas)
    print("  ||b - A x||_inf = %.3e" % res)
    print("  ||x_gauss - x_numpy||_inf = %.3e" % dif)
    print("  alunos por college:", alunos, " custo total = %.2f" % total)

    # ---- tabelas e valores para o relatorio
    linhas = []
    for c, xi, xn, xc in zip(colleges, x, x_np, x_cent):
        linhas.append([c, num(xi, casas=6), num(xn, casas=6), num(xc, casas=2)])
    salvar_tabela_tex("ex1_solucao.tex", "lrrr",
                      ["College", r"$x_i$ (Gauss)",
                       r"$x_i$ (\texttt{np.linalg.solve})",
                       r"Custo (US\$)"], linhas)
    salvar_csv("ex1_solucao.csv", ["college", "x_gauss", "x_numpy", "custo_centavos"],
               [[c, xi, xn, xc] for c, xi, xn, xc in zip(colleges, x, x_np, x_cent)])

    mat = lambda M: r" \\ ".join(" & ".join(num(v, casas=0) for v in lin) for lin in M)
    salvar_valores("ex1_valores.tex", {
        "exUmA": mat(A),
        "exUmb": r" \\ ".join(num(v, casas=0) for v in B),
        "exUmxSci": num(x_cent[0], casas=2),
        "exUmxEng": num(x_cent[1], casas=2),
        "exUmxCS": num(x_cent[2], casas=2),
        "exUmTrocas": str(trocas),
        "exUmRes": num(res, sci=1),
        "exUmResRel": num(res / norma_inf(B), sci=1),
        "exUmDif": num(dif, sci=1),
        "exUmAlunosSci": num(alunos[0], casas=0),
        "exUmAlunosEng": num(alunos[1], casas=0),
        "exUmAlunosCS": num(alunos[2], casas=0),
        "exUmTotal": num(total, casas=2),
    })
    print("arquivos gravados em resultados/")


if __name__ == "__main__":
    main()
