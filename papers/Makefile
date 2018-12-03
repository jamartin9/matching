.PHONY: clean all

all: paper.pdf

results.eps: results.dat results.gnuplot
	gnuplot results.gnuplot

results.pdf: results.eps
	epspdf results.eps

paper.pdf: results.pdf paper.tex
	pdflatex paper.tex

clean:
	rm -f paper.pdf results.pdf results.eps *.log *.aux
