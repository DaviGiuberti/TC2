"""
Parte 2 - Problema 1: rede de aducao por gravidade.

Incognitas x = (Q1, Q2, Q3, H). Partimos do esqueleto do enunciado e
completamos a Jacobiana. Itens (a) a (f) e, no final, uma conferencia
independente com scipy.optimize.fsolve (so conferencia, nao e o metodo).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metodos import (erros_e_ordem, jacobiana_discreta, newton, norma_inf, num,
                     pasta_resultados, salvar_csv, salvar_tabela_tex,
                     salvar_valores, solucao_referencia)

# ---------------------------------------------------------------- esqueleto
g = 9.81
z = np.array([100.0, 85.0, 60.0])
L = np.array([1200.0, 900.0, 1500.0])
D = np.array([0.30, 0.25, 0.25])
f = np.array([0.022, 0.024, 0.024])
K = 8*f*L/(np.pi**2*g*D**5)
q = 0.200


def F(x, K=K, q=q):
    Q, H = x[:3], x[3]
    r = np.empty(4)
    r[:3] = K*Q*np.abs(Q) + H - z      # Bernoulli em cada adutora
    r[3] = Q.sum() - q                 # continuidade no no J
    return r


# <<jacobiana>>
def J(x, K=K):
    # d/dQ (K Q|Q|) = 2 K |Q|  ->  diagonal; coluna de H e linha da
    # continuidade formam a "seta"
    Q = x[:3]
    M = np.zeros((4, 4))
    M[0, 0], M[1, 1], M[2, 2] = 2*K*np.abs(Q)
    M[:3, 3] = 1.0          # dF_i/dH = 1, i = 1, 2, 3
    M[3, :3] = 1.0          # dF_4/dQ_i = 1
    return M
# <<fim>>


def det_seta(x, K=K):
    """det J = -(d1 d2 + d1 d3 + d2 d3), d_i = 2 K_i |Q_i| (item c)."""
    d = 2*K*np.abs(x[:3])
    return -(d[0]*d[1] + d[0]*d[2] + d[1]*d[2])


# item (e): Q3 = 0 fixo e q vira incognita, y = (Q1, Q2, H, q)
def G(y, K=K):
    Q1, Q2, H, qq = y
    Q = np.array([Q1, Q2, 0.0])
    r = np.empty(4)
    r[:3] = K*Q*np.abs(Q) + H - z
    r[3] = Q1 + Q2 - qq
    return r


def JG(y, K=K):
    Q1, Q2 = y[0], y[1]
    return np.array([[2*K[0]*abs(Q1), 0.0, 1.0, 0.0],
                     [0.0, 2*K[1]*abs(Q2), 1.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0],
                     [1.0, 1.0, 0.0, -1.0]])


# ---------------------------------------------------------------- parametros
TOL = 1e-10       # ||s^(k)||_inf
TOL_F = 1e-10     # ||F(x^(k))||_inf  (slide 25)
KMAX = 50
x0_b = np.array([0.10, 0.10, -0.05, 70.0])
x0_c = np.array([0.0, 0.0, 0.0, 70.0])
f1_velho = 0.030

AZUL, LARANJA = "#2a78d6", "#eb6834"
TINTA, TINTA2 = "#0b0b0b", "#52514e"

NOME_PARADA = {"passo": r"$\|s^{(k)}\|_\infty$",
               "residuo": r"$\|F(x^{(k)})\|_\infty$",
               "kmax": "n\\'umero m\\'aximo de itera\\c{c}\\~oes"}


def tabela_newton(nome, hist, xstar, rotulos, casas):
    e, expo, p = erros_e_ordem(hist, xstar)
    linhas, lcsv = [], []
    for h, ek, xk, pk in zip(hist, e, expo, p):
        linhas.append([str(h["k"])] +
                      [num(v, casas=c) for v, c in zip(h["x"], casas)] +
                      [num(h["normaF"], sci=1), num(ek, sci=1),
                       "" if xk is None else "$%d$" % xk, num(pk, casas=2)])
        lcsv.append([h["k"], *h["x"], h["normaF"], h["normaS"], ek, pk])
    cab = [r"$k$"] + rotulos + [r"$\|F(x^{(k)})\|_\infty$",
                                r"$\|x^{(k)}-x^*\|_\infty$", "exp.", r"$p_k$"]
    salvar_tabela_tex(nome + ".tex", "r" * len(cab), cab, linhas)
    salvar_csv(nome + ".csv", ["k"] + [r.strip("$") for r in rotulos] +
               ["normaF", "normaS", "erro", "p"], lcsv)
    return e, expo, p


def resolve(Ff, Jf, x0, nome, rotulos, casas):
    x, hist, parada = newton(Ff, Jf, x0, TOL, KMAX, TOL_F)
    xstar = solucao_referencia(Ff, Jf, x)
    e, expo, p = tabela_newton(nome, hist, xstar, rotulos, casas)
    print("  %s: %d iteracoes, parada por %s" % (nome, hist[-1]["k"], parada))
    for h, ek, pk in zip(hist, e, p):
        print("    k=%d x=%s |F|=%.2e e=%.2e p=%s" % (h["k"], np.array2string(h["x"], precision=10),
                                                    h["normaF"], ek, "%.2f" % pk if pk == pk else "-"))
    return x, hist, parada, e, expo, p


def main():
    V = {}          # valores que vao para o relatorio
    print("Parte 2 - rede de aducao")
    V["pdTol"] = num(TOL, sci=0)
    V["pdTolF"] = num(TOL_F, sci=0)
    V["pdKmax"] = str(KMAX)

    # ================================================================ (a)
    print("(a) K =", K)
    linhas = [[str(i + 1), num(f[i], casas=3), num(L[i], casas=0), num(D[i], casas=2),
               num(K[i], casas=4)] for i in range(3)]
    salvar_tabela_tex("p2a_K.tex", "crrrr",
                      [r"Adutora $i$", r"$f_i$", r"$L_i$ (\si{m})", r"$D_i$ (\si{m})",
                       r"$K_i$ (\si{s^2/m^5})"], linhas)
    for i, nome in enumerate(("Ki", "Kii", "Kiii")):
        V["pd" + nome] = num(K[i], casas=2)
    # conferencia da Jacobiana analitica com o Newton discretizado (slide 20)
    difs = []
    for nome_pt, pt in (("x0", x0_b),):
        Ja, Jh = J(pt), jacobiana_discreta(F, pt, h=1e-8)
        difs.append(norma_inf((Ja - Jh).ravel()))
        print("  max|J - J_h| em %s = %.2e" % (nome_pt, difs[-1]))
        print("  J(x0) =\n", Ja)
    V["pdDifJxz"] = num(difs[0], sci=1)
    # erro de truncamento esperado da diferenca progressiva em K Q|Q|:
    # (h/2) * |d2/dQ2 (K Q|Q|)| = (h/2) * 2K = K h  (o maior e o de K3)
    V["pdKiiih"] = num(K[2] * 1e-8, sci=1)
    Ja0 = J(x0_b)
    V["pdJxzDi"] = num(Ja0[0, 0], casas=4)
    V["pdJxzDii"] = num(Ja0[1, 1], casas=4)
    V["pdJxzDiii"] = num(Ja0[2, 2], casas=4)

    # ================================================================ (b)
    print("(b) Newton a partir de x0 =", x0_b)
    rot = [r"$Q_1^{(k)}$", r"$Q_2^{(k)}$", r"$Q_3^{(k)}$", r"$H^{(k)}$"]
    xb, hb, pb, eb, expb, ppb = resolve(F, J, x0_b, "p2b_iteracoes", rot, [10, 10, 10, 8])
    Jxs = J(xb)
    difJ = norma_inf((Jxs - jacobiana_discreta(F, xb, h=1e-8)).ravel())
    print("  max|J - J_h| em x* = %.2e" % difJ)
    V["pdDifJxs"] = num(difJ, sci=1)
    V["pdbIt"] = str(hb[-1]["k"])
    V["pdbParada"] = NOME_PARADA[pb]
    V["pdQi"] = num(xb[0], casas=6)
    V["pdQii"] = num(xb[1], casas=6)
    V["pdQiii"] = num(xb[2], casas=6)
    V["pdH"] = num(xb[3], casas=4)
    V["pdQiiiAbs"] = num(abs(xb[2]), casas=6)
    V["pdQiiiLs"] = num(abs(xb[2]) * 1000, casas=1)
    V["pdResFinal"] = num(hb[-1]["normaF"], sci=1)
    V["pdbExpoentes"] = "$" + r" \to ".join("%d" % v for v in expb if v is not None) + "$"
    ps = [v for v in ppb if v == v]
    V["pdbPs"] = ", ".join(num(v, casas=2) for v in ps)
    V["pdbPUlt"] = num(ps[-1], casas=2)
    V["pdDetSol"] = num(det_seta(xb), casas=2)
    V["pdDetSolNp"] = num(np.linalg.det(Jxs), casas=2)
    # a primeira iteracao ja satisfaz a continuidade (equacao linear)
    V["pdbContUm"] = num(abs(hb[1]["x"][:3].sum() - q), sci=1)
    V["pdbHum"] = num(hb[1]["x"][3], casas=2)
    V["pdbEzero"] = num(eb[0], casas=2)
    V["pdbEum"] = num(eb[1], casas=2)

    # ================================================================ (c)
    print("(c) x0 =", x0_c)
    J0 = J(x0_c)
    det_np = np.linalg.det(J0)
    posto = np.linalg.matrix_rank(J0)
    print("  J(x0) =\n", J0)
    print("  det (numpy) =", det_np, " det (formula) =", det_seta(x0_c), " posto =", posto)
    try:
        newton(F, J, x0_c, TOL, KMAX, TOL_F)
        msg_c = "convergiu"
    except np.linalg.LinAlgError as err:
        msg_c = "LinAlgError: " + str(err)
    print("  newton:", msg_c)
    V["pdcDet"] = num(det_np, casas=1)
    V["pdcPosto"] = str(posto)
    V["pdcErro"] = r"\texttt{numpy.linalg.LinAlgError: %s}" % msg_c.split(": ", 1)[1]

    # outros pontos: formula x numpy e o que o Newton faz
    pontos = [("$(0;\\,0;\\,0;\\,70)$", x0_c),
              ("$(0;\\,0;\\,0{,}1;\\,70)$", np.array([0.0, 0.0, 0.1, 70.0])),
              ("$(0{,}2;\\,0;\\,0;\\,50)$", np.array([0.2, 0.0, 0.0, 50.0])),
              ("$(0{,}1;\\,0{,}1;\\,0;\\,70)$", np.array([0.1, 0.1, 0.0, 70.0])),
              ("$(10^{-3};\\,10^{-3};\\,10^{-3};\\,70)$", np.array([1e-3, 1e-3, 1e-3, 70.0])),
              ("$x^{(0)}$ do item (b)", x0_b)]
    linhas, lcsv = [], []
    for rotulo, pt in pontos:
        dform, dnp = det_seta(pt), np.linalg.det(J(pt))
        try:
            xs, hs, ps_ = newton(F, J, pt, TOL, KMAX, TOL_F)
            ok = norma_inf(xs - xb) < 1e-8
            passo1 = hs[0]["normaS"]
            res = "converge (%d it.)" % hs[-1]["k"] if ok else "outra raiz?"
            res_tex = "converge em %d it." % hs[-1]["k"] if ok else "n\\~ao converge para $x^*$"
            p1 = num(passo1, sci=1)
        except np.linalg.LinAlgError as err:
            res, res_tex, p1 = "LinAlgError: " + str(err), r"\texttt{LinAlgError}", ""
        print("  ponto %s: det formula = %.4e, numpy = %.4e -> %s" % (pt, dform, dnp, res))
        linhas.append([rotulo, num(dform, sci=2), num(dnp, sci=2), p1, res_tex])
        lcsv.append([str(pt), dform, dnp, res])
    salvar_tabela_tex("p2c_pontos.tex", "lrrrl",
                      [r"Ponto $x^{(0)}$", r"$\det J$ (f\'ormula)", r"$\det J$ (\texttt{numpy})",
                       r"$\|s^{(0)}\|_\infty$", "Newton"], linhas)
    salvar_csv("p2c_pontos.csv", ["ponto", "det_formula", "det_numpy", "newton"], lcsv)
    # passo gigante perto do ponto singular
    _, hq, _ = newton(F, J, np.array([1e-3, 1e-3, 1e-3, 70.0]), TOL, KMAX, TOL_F)
    V["pdcPertoIt"] = str(hq[-1]["k"])
    V["pdcPertoPasso"] = num(hq[0]["normaS"], sci=1)
    V["pdcPertoQ"] = num(norma_inf(hq[1]["x"][:3]), casas=1)

    # ================================================================ (d)
    print("(d) sinais e velocidades")
    area = np.pi * D**2 / 4
    v = xb[:3] / area
    linhas = []
    for i in range(3):
        sentido = "reservat\\'orio $\\to$ n\\'o" if xb[i] > 0 else "n\\'o $\\to$ reservat\\'orio"
        dentro = "sim" if 0.6 <= abs(v[i]) <= 3.0 else "n\\~ao"
        linhas.append([str(i + 1), num(xb[i], casas=6), sentido, num(area[i], casas=6),
                       num(v[i], casas=3), dentro])
        print("  adutora %d: Q = %.6f, A = %.6f, v = %.3f m/s" % (i + 1, xb[i], area[i], v[i]))
    salvar_tabela_tex("p2d_velocidades.tex", "crlrrc",
                      [r"$i$", r"$Q_i$ (\si{m^3/s})", "Sentido", r"$A_i$ (\si{m^2})",
                       r"$v_i$ (\si{m/s})", r"Na faixa?"], linhas)
    for i, nome in enumerate(("i", "ii", "iii")):
        V["pdv" + nome] = num(abs(v[i]), casas=2)
    V["pdEntraNo"] = num(xb[0] + xb[1], casas=6)
    V["pdHmenosZiii"] = num(xb[3] - z[2], casas=4)

    # ================================================================ (e)
    print("(e) demanda critica q* (Q3 = 0)")
    y0 = np.array([xb[0], xb[1], xb[3], q])     # parte da solucao do item (b)
    rot_e = [r"$Q_1^{(k)}$", r"$Q_2^{(k)}$", r"$H^{(k)}$", r"$q^{(k)}$"]
    ye, he, pe, ee, expe, ppe = resolve(G, JG, y0, "p2e_iteracoes", rot_e, [10, 10, 8, 10])
    qc = ye[3]
    # confere: q* no sistema original tem que dar Q3 = 0
    xq, hq_, _ = newton(lambda x: F(x, q=qc), J, x0_b, TOL, KMAX, TOL_F)
    print("  q* = %.10f ; no sistema original: Q3 = %.3e, H = %.10f" % (qc, xq[2], xq[3]))
    V["pdeIt"] = str(he[-1]["k"])
    V["pdeParada"] = NOME_PARADA[pe]
    V["pdqc"] = num(qc, casas=6)
    V["pdqcLs"] = num(qc * 1000, casas=1)
    V["pdeQi"] = num(ye[0], casas=6)
    V["pdeQii"] = num(ye[1], casas=6)
    V["pdeH"] = num(ye[2], casas=4)
    V["pdeQiiiConf"] = num(xq[2], sci=1)
    V["pdeHConf"] = num(xq[3], casas=10)
    V["pdeConfIt"] = str(hq_[-1]["k"])
    V["pdeDetJG"] = num(-(2*K[0]*abs(ye[0]))*(2*K[1]*abs(ye[1])), casas=1)

    # ================================================================ (f)
    print("(f) f1 = %.3f" % f1_velho)
    f_v = f.copy()
    f_v[0] = f1_velho
    K_v = 8*f_v*L/(np.pi**2*g*D**5)
    Fv = lambda x: F(x, K=K_v)
    Jv = lambda x: J(x, K=K_v)
    xf, hf, pf, ef, expf, ppf = resolve(Fv, Jv, x0_b, "p2f_iteracoes", rot, [10, 10, 10, 8])
    yf, hfe, _, _, _, _ = resolve(lambda y: G(y, K=K_v), lambda y: JG(y, K=K_v),
                                  np.array([xf[0], xf[1], xf[3], q]), "p2f_qc_iteracoes",
                                  rot_e, [10, 10, 8, 10])
    qc_v = yf[3]
    grandezas = [("$Q_1$ (\\si{m^3/s})", xb[0], xf[0], 6),
                 ("$Q_2$ (\\si{m^3/s})", xb[1], xf[1], 6),
                 ("$Q_3$ (\\si{m^3/s})", xb[2], xf[2], 6),
                 ("$H$ (\\si{m})", xb[3], xf[3], 4),
                 ("$q^*$ (\\si{m^3/s})", qc, qc_v, 6)]
    linhas, lcsv = [], []
    varia = {}
    for rotulo, a, bv, c in grandezas:
        dv = 100 * (bv - a) / a
        varia[rotulo] = dv
        linhas.append([rotulo, num(a, casas=c), num(bv, casas=c), num(dv, casas=2)])
        lcsv.append([rotulo, a, bv, dv])
        print("  %-22s %.6f -> %.6f  (%+.2f %%)" % (rotulo, a, bv, dv))
    salvar_tabela_tex("p2f_sensibilidade.tex", "lrrr",
                      ["Grandeza", r"$f_1 = \num{0.022}$", r"$f_1 = \num{0.030}$",
                       r"Varia\c{c}\~ao (\%)"], linhas, meio=[4])
    salvar_csv("p2f_sensibilidade.csv", ["grandeza", "f1_0022", "f1_0030", "variacao_pct"], lcsv)
    V["pdfKi"] = num(K_v[0], casas=2)
    V["pdfKiVar"] = num(100 * (K_v[0] - K[0]) / K[0], casas=2)
    V["pdfIt"] = str(hf[-1]["k"])
    V["pdfQi"] = num(xf[0], casas=6)
    V["pdfQii"] = num(xf[1], casas=6)
    V["pdfQiii"] = num(xf[2], casas=6)
    V["pdfH"] = num(xf[3], casas=4)
    V["pdfqc"] = num(qc_v, casas=6)
    V["pdfVarQi"] = num(varia[grandezas[0][0]], casas=2)
    V["pdfVarQii"] = num(varia[grandezas[1][0]], casas=2)
    V["pdfVarQiii"] = num(varia[grandezas[2][0]], casas=2)
    V["pdfVarH"] = num(varia[grandezas[3][0]], casas=2)
    V["pdfVarqc"] = num(varia[grandezas[4][0]], casas=2)
    # o mesmo com a carga medida a partir do reservatorio 3 (H - z3)
    V["pdfHzAntes"] = num(xb[3] - z[2], casas=4)
    V["pdfHzDepois"] = num(xf[3] - z[2], casas=4)
    V["pdfVarHz"] = num(100 * ((xf[3] - z[2]) - (xb[3] - z[2])) / (xb[3] - z[2]), casas=2)
    V["pdfvi"] = num(xf[0] / area[0], casas=2)
    V["pdfQiiiLs"] = num(abs(xf[2]) * 1000, casas=1)
    V["pdfDeltaQiLs"] = num(abs(xf[0] - xb[0]) * 1000, casas=1)
    V["pdfDeltaqcLs"] = num(abs(qc_v - qc) * 1000, casas=1)
    V["pdfqcLs"] = num(qc_v * 1000, casas=1)

    # ================================================================ figura
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for e_, cor, mk, rotulo in ((eb, AZUL, "o", r"$f_1 = 0{,}022$ (item b)"),
                                (ef, LARANJA, "s", r"$f_1 = 0{,}030$ (item f)")):
        ks = np.arange(len(e_))
        ep = np.array(e_)
        m = ep > 0
        ax.semilogy(ks[m], ep[m], color=cor, lw=2, marker=mk, ms=6, label=rotulo)
    ax.set_xlabel("iteração $k$", color=TINTA2)
    ax.set_ylabel(r"$\|x^{(k)} - x^*\|_\infty$", color=TINTA2)
    ax.grid(True, which="major", color="#e4e3df", lw=0.6)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#b5b4ae")
    ax.tick_params(colors=TINTA2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(pasta_resultados() / "p2_convergencia.pdf")
    plt.close(fig)

    # ================================================================ fsolve
    print("Conferencia com scipy.optimize.fsolve")
    try:
        from scipy.optimize import fsolve
        casos = [("item (b)", "item (b)", F, x0_b, xb),
                 ("item (e)", "item (e)", G, y0, ye),
                 ("item (f) solucao", "item (f), solu\\c{c}\\~ao", Fv, x0_b, xf),
                 ("item (f) q*", "item (f), $q^*$", lambda y: G(y, K=K_v),
                  np.array([xf[0], xf[1], xf[3], q]), yf)]
        linhas, lcsv = [], []
        for rot_txt, rotulo, Ff, xi, xn in casos:
            # sem fprime: o fsolve aproxima a Jacobiana sozinho, assim a
            # conferencia nao depende da nossa J
            xs, info, ier, mesg = fsolve(Ff, xi, full_output=True, xtol=1e-13)
            dif = norma_inf(xs - xn)
            print("  %s: ier=%d, max|x_fsolve - x_newton| = %.2e, nfev=%d"
                  % (rot_txt, ier, dif, info["nfev"]))
            linhas.append([rotulo, str(ier), str(info["nfev"]), num(dif, sci=1)])
            lcsv.append([rot_txt, ier, info["nfev"], dif])
        salvar_tabela_tex("p2_fsolve.tex", "lrrr",
                          ["Caso", r"\texttt{ier}", r"Avalia\c{c}\~oes de $F$",
                           r"$\|x_{\mathrm{fsolve}} - x_{\mathrm{Newton}}\|_\infty$"], linhas)
        salvar_csv("p2_fsolve.csv", ["caso", "ier", "nfev", "dif"], lcsv)
        V["pdFsolveMax"] = num(max(r[3] for r in lcsv), sci=1)
    except ImportError:
        print("  scipy nao instalado: conferencia pulada")
        salvar_tabela_tex("p2_fsolve.tex", "l", ["scipy n\\~ao instalado"], [])
        V["pdFsolveMax"] = "(scipy ausente)"

    salvar_valores("p2_valores.tex", V)
    print("arquivos gravados em resultados/")


if __name__ == "__main__":
    main()
