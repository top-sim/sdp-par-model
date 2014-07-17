
EMAXINPUTS = MajCycleModel.tex

default: $(EMAXINPUTS:.tex=.pdf)

%.bbl: %.aux
	bibtex $*

%.pdf: %.tex %.bbl
	makeindex $*.nlo -s nomencl.ist -o $*.nls
	pdflatex -shell-escape -interaction=nonstopmode "\input" $<
