"""
metodos.py - funcoes reutilizaveis do TC2 de Calculo Numerico
(prof. Leduino, ICT-UNIFESP).

De onde veio cada rotina (citado no relatorio):

  norma_inf ............ "Normas Vetoriais", slides 7 e 8 (norma do maximo)
  retro_substituicao ... "Sistemas Lineares", slide 7
  gauss_pivoteamento ... "Sistemas Lineares", slides 14 e 18
  jacobi ............... "Sistemas Lineares", slide 23 (parada do slide 24)
  gauss_seidel ......... "Sistemas Lineares", slide 27
  criterio_linhas ...... "Sistemas Lineares", slide 30
  sassenfeld ........... "Sistemas Lineares", slide 31
  newton ............... codigo do livro da UFRGS (REAMAT, secao 5.1.1) que
                         veio no enunciado da parte 1, adaptado com os slides
                         9, 12, 13 e 25 da aula de sistemas nao-lineares
  jacobiana_discreta ... aula de sistemas nao-lineares, slide 20
  expoentes/ordem ...... aula de sistemas nao-lineares, slides 16 e 17

No fim do arquivo ficam funcoes pequenas para gravar tabelas (.csv e .tex)
e os numeros que o relatorio usa, para nada ser digitado a mao no .tex.
"""

import csv
import math
from pathlib import Path

import numpy as np

EPS = np.finfo(float).eps


# ---------------------------------------------------------------------------
# Normas
# ---------------------------------------------------------------------------

def norma_inf(v):
    """||v||_inf = max |v_i| (slides de normas, slide 7)."""
    return float(np.max(np.abs(v)))


# ---------------------------------------------------------------------------
# Sistemas lineares: metodos diretos
# ---------------------------------------------------------------------------

def retro_substituicao(U, c):
    """Resolve Ux = c com U triangular superior (slide 7)."""
    n = len(c)
    x = np.zeros(n)
    x[n - 1] = c[n - 1] / U[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s = s + U[i, j] * x[j]
        x[i] = (c[i] - s) / U[i, i]
    return x


def gauss_pivoteamento(A, b, eps=1e-12):
    """Eliminacao de Gauss com pivotamento parcial (slides 14 e 18) seguida
    de retro-substituicao. Trabalha em copias de A e b.

    Retorna x e o numero de trocas de linha feitas."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    trocas = 0
    for k in range(n - 1):
        # 1) linha do maior pivo em modulo
        p = k + int(np.argmax(np.abs(A[k:, k])))
        # 2) troca as linhas p e k (em A e em b)
        if p != k:
            A[[k, p]] = A[[p, k]]
            b[[k, p]] = b[[p, k]]
            trocas += 1
        if abs(A[k, k]) < eps:
            raise np.linalg.LinAlgError("matriz singular (pivo nulo)")
        # 3) elimina normalmente
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k] = 0.0
            A[i, k + 1:] = A[i, k + 1:] - m * A[k, k + 1:]
            b[i] = b[i] - m * b[k]
    if abs(A[n - 1, n - 1]) < eps:
        raise np.linalg.LinAlgError("matriz singular (pivo nulo)")
    return retro_substituicao(A, b), trocas


# ---------------------------------------------------------------------------
# Sistemas lineares: metodos iterativos
# ---------------------------------------------------------------------------

def jacobi(A, b, x0, tol, kmax):
    """Gauss-Jacobi (slide 23). Criterio de parada: erro relativo
    d = max|xnovo - x| / max|xnovo| < tol (slide 24).

    Retorna x, historico [(k, x^(k), d_k)] e o motivo da parada."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(b)
    x = np.array(x0, dtype=float)
    hist = [(0, x.copy(), np.nan)]
    for k in range(1, kmax + 1):
        xnovo = np.zeros(n)
        for i in range(n):
            s = 0.0
            for j in range(n):
                if j != i:
                    s = s + A[i, j] * x[j]      # x ANTIGO
            xnovo[i] = (b[i] - s) / A[i, i]
        d = norma_inf(xnovo - x) / norma_inf(xnovo)
        x = xnovo                               # so agora atualiza
        hist.append((k, x.copy(), d))
        if d < tol:
            return x, hist, "tolerancia"
    return x, hist, "kmax"


def gauss_seidel(A, b, x0, tol, kmax):
    """Gauss-Seidel (slide 27). Mesmo criterio de parada do Jacobi."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(b)
    x = np.array(x0, dtype=float)
    hist = [(0, x.copy(), np.nan)]
    for k in range(1, kmax + 1):
        xant = x.copy()                         # copia para o teste
        for i in range(n):
            s = 0.0
            for j in range(n):
                if j != i:
                    s = s + A[i, j] * x[j]      # JA ATUALIZADO
            x[i] = (b[i] - s) / A[i, i]
        d = norma_inf(x - xant) / norma_inf(x)
        hist.append((k, x.copy(), d))
        if d < tol:
            return x, hist, "tolerancia"
    return x, hist, "kmax"


def criterio_linhas(A):
    """alpha_i = sum_{j!=i} |a_ij| / |a_ii|  (slide 30)."""
    A = np.abs(np.asarray(A, dtype=float))
    d = np.diag(A)
    alfas = (A.sum(axis=1) - d) / d
    return alfas, float(alfas.max())


def sassenfeld(A):
    """beta_i do criterio de Sassenfeld, calculados de beta_1 ate beta_n
    (slide 31)."""
    A = np.abs(np.asarray(A, dtype=float))
    n = A.shape[0]
    betas = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += A[i, j] * betas[j]
        for j in range(i + 1, n):
            s += A[i, j]
        betas[i] = s / A[i, i]
    return betas, float(betas.max())


def matrizes_iteracao(A):
    """Matrizes C de x^(k+1) = C x^(k) + g para Jacobi e Seidel
    (slides 21 e 27): C_J = -D^{-1}(L+U) e C_GS = -(D+L)^{-1} U."""
    A = np.asarray(A, dtype=float)
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    CJ = -np.linalg.solve(D, L + U)
    CGS = -np.linalg.solve(D + L, U)
    return CJ, CGS


def raio_espectral(C):
    return float(np.max(np.abs(np.linalg.eigvals(C))))


# ---------------------------------------------------------------------------
# Sistemas nao-lineares: metodo de Newton
# ---------------------------------------------------------------------------

def newton(F, JF, x0, TOL, N, TOL_F=None):
    """Metodo de Newton para F(x) = 0.

    Partimos do codigo do livro da UFRGS (enunciado da parte 1) e mudamos:
      * o passo s^(k) sai de J(x^(k)) s = -F(x^(k)) com np.linalg.solve
        (LU com pivotamento) em vez de -inv(J).dot(F): slide 13, nunca
        calcular J^{-1};
      * guardamos o historico (x^(k), ||F(x^(k))||_inf, ||s^(k)||_inf);
      * alem de ||s^(k)||_inf < TOL (UFRGS e slide 12) tambem paramos pelo
        residuo ||F(x^(k))||_inf < TOL_F e pelo maximo de iteracoes N
        (slide 25). No caso de N, em vez de levantar NameError como no
        livro, devolvemos o que temos com o aviso "kmax".

    Se J(x^(k)) for singular, np.linalg.solve levanta LinAlgError; quem
    chama decide o que fazer.

    Retorna (x, hist, parada). hist[k] = dict com k, x, normaF, normaS
    (normaS = nan na ultima linha, onde nao se calcula mais passo)."""
    # preliminares
    x = np.copy(x0).astype('double')
    hist = []
    k = 0
    # iteracoes
    while (k < N):
        Fx = F(x)
        nF = norma_inf(Fx)
        # criterio de parada pelo residuo (slide 25)
        if TOL_F is not None and nF < TOL_F:
            hist.append(dict(k=k, x=x.copy(), normaF=nF, normaS=np.nan))
            return x, hist, "residuo"
        # iteracao Newton: J s = -F  (antes: delta = -inv(JF(x)).dot(F(x)))
        s = np.linalg.solve(JF(x), -Fx)
        hist.append(dict(k=k, x=x.copy(), normaF=nF, normaS=norma_inf(s)))
        x = x + s
        k += 1
        # criterio de parada pelo passo (UFRGS / slide 12)
        if (norma_inf(s) < TOL):
            hist.append(dict(k=k, x=x.copy(), normaF=norma_inf(F(x)),
                             normaS=np.nan))
            return x, hist, "passo"
    # maximo de iteracoes (slide 25)
    hist.append(dict(k=k, x=x.copy(), normaF=norma_inf(F(x)), normaS=np.nan))
    print("  aviso: newton parou pelo numero maximo de iteracoes")
    return x, hist, "kmax"


def solucao_referencia(F, JF, x0, kmax=100):
    """x* usado para medir ||x^(k) - x*||: Newton continuado ate o passo
    ficar no nivel do epsilon de maquina (||s|| <= 4 eps ||x||)."""
    x = np.array(x0, dtype=float)
    for _ in range(kmax):
        s = np.linalg.solve(JF(x), -F(x))
        x = x + s
        if norma_inf(s) <= 4 * EPS * max(1.0, norma_inf(x)):
            break
    # mais um passo para "assentar" no ultimo bit
    return x + np.linalg.solve(JF(x), -F(x))


def jacobiana_discreta(F, x, h=1e-8):
    """Newton discretizado (slide 20): J_ij ~ (f_i(x + h e_j) - f_i(x)) / h."""
    x = np.asarray(x, dtype=float)
    Fx = F(x)
    n = len(x)
    J = np.zeros((len(Fx), n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        J[:, j] = (F(x + e) - Fx) / h
    return J


def erros_e_ordem(hist, xstar):
    """Para cada iterado: e_k = ||x^(k) - x*||_inf, o expoente
    floor(log10 e_k) (como no slide 17) e a estimativa da ordem
        p_k = log(e_k / e_{k-1}) / log(e_{k-1} / e_{k-2}).
    p_k so e calculado quando o erro caiu nos dois passos (e_{k-2} > e_{k-1}
    > e_k) e enquanto e_k esta acima do ruido de arredondamento
    (10 eps ||x*||); fora disso o numero nao quer dizer nada."""
    e = [norma_inf(h["x"] - xstar) for h in hist]
    ruido = 10 * EPS * max(1.0, norma_inf(xstar))
    expo = [math.floor(math.log10(v)) if v > 0 else None for v in e]
    p = [np.nan] * len(e)
    for k in range(2, len(e)):
        a, b_, c = e[k - 2], e[k - 1], e[k]
        if c > ruido and a > b_ > c:
            p[k] = math.log(c / b_) / math.log(b_ / a)
    return e, expo, p


# ---------------------------------------------------------------------------
# Saida: tabelas e valores para o relatorio
# ---------------------------------------------------------------------------

RAIZ = Path(__file__).resolve().parent.parent
RESULTADOS = RAIZ / "resultados"


def pasta_resultados():
    RESULTADOS.mkdir(exist_ok=True)
    return RESULTADOS


def num(v, casas=None, sci=None):
    """Numero no formato do siunitx: \\num{...}. Com casas=d usa d casas
    decimais; com sci=d usa notacao cientifica com d casas."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if sci is not None:
        if v == 0:
            return r"\num{0}"
        return r"\num{%.*e}" % (sci, v)
    if casas is not None:
        s = "%.*f" % (casas, v)
        if float(s) == 0:          # evita "-0,000"
            s = s.lstrip("-")
        return r"\num{%s}" % s
    return r"\num{%s}" % repr(v)


def salvar_csv(nome, cabecalho, linhas):
    caminho = pasta_resultados() / nome
    with open(caminho, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cabecalho)
        for linha in linhas:
            w.writerow(linha)
    return caminho


def salvar_tabela_tex(nome, colunas, cabecalho, linhas, meio=None):
    """Grava so o ambiente tabular (booktabs). O relatorio faz o \\input
    dentro de um table com legenda. 'meio' = indices de linhas antes das
    quais entra um \\midrule extra."""
    meio = set(meio or [])
    out = [r"\begin{tabular}{%s}" % colunas, r"\toprule",
           " & ".join(cabecalho) + r" \\", r"\midrule"]
    for i, linha in enumerate(linhas):
        if i in meio:
            out.append(r"\midrule")
        out.append(" & ".join(linha) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", ""]
    caminho = pasta_resultados() / nome
    caminho.write_text("\n".join(out), encoding="utf-8")
    return caminho


def salvar_valores(nome, valores):
    """Grava \\newcommand{\\nome}{valor} para cada item. Os nomes so podem
    ter letras (regra do LaTeX)."""
    out = ["% gerado automaticamente por src/ - nao editar a mao"]
    for chave, valor in valores.items():
        if not chave.isalpha():
            raise ValueError("nome de macro invalido: " + chave)
        out.append(r"\newcommand{\%s}{%s}" % (chave, valor))
    caminho = pasta_resultados() / nome
    caminho.write_text("\n".join(out) + "\n", encoding="utf-8")
    return caminho
