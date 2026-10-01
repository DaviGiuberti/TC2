# TC2 de Cálculo Numérico (partes 1 e 2)

Trabalho Computacional 2 da disciplina de Cálculo Numérico, prof. Luiz Leduino
de Salles Neto, ICT/UNIFESP. Entrega: 02/10/2026.

## Integrantes

| Nome | RA |
|---|---|
| Gabriel Lemos da Silva | 140971 |
| Davi Giuberti de Souza | 168631 |
| Rodrigo Ezequiel Silva dos Anjos | 188215 |
| Mayara Martinez Boaro | 188197 |

## Estrutura

```
src/
  metodos.py        funções reutilizáveis: normas, Gauss com pivotamento,
                    Jacobi, Gauss-Seidel, critérios de convergência,
                    Newton (adaptado do livro da UFRGS), Newton discretizado
  parte1_ex1.py     custo anual por aluno (sistema 3x3, Gauss)
  parte1_ex2.py     rede de resistores (Jacobi x Gauss-Seidel x Gauss)
  parte1_ex3.py     parábola x elipse (Newton, 2 raízes)
  parte1_ex4.py     sistema 3x3 não-linear perto da origem (Newton)
  parte2_aducao.py  rede de adução por gravidade, itens (a) a (f) + fsolve
resultados/         tabelas (.csv e .tex), figuras (.pdf) e valores (*_valores.tex)
relatorio/          relatorio.tex e relatorio.pdf
```

## Como rodar

Python 3.10 ou mais novo.

```bash
pip install -r requirements.txt

python src/parte1_ex1.py
python src/parte1_ex2.py
python src/parte1_ex3.py
python src/parte1_ex4.py
python src/parte2_aducao.py
```

Cada script roda sozinho, de qualquer pasta, e grava as saídas em
`resultados/`. O `scipy` só é usado na conferência final da parte 2 com
`fsolve`; sem ele o script avisa e pula essa parte.

## Relatório

O relatório lê as tabelas, figuras e números direto de `resultados/`, então
os scripts precisam rodar antes de compilar:

```bash
cd relatorio
latexmk -pdf relatorio.tex      # ou: pdflatex relatorio.tex (duas vezes)
```

Pacotes LaTeX usados: babel (brazil), siunitx, booktabs, listings, xcolor,
geometry, caption, float, hyperref, lmodern.
