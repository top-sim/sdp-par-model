
EMAXINPUTS = MajCycleModel.tex

default: $(EMAXINPUTS:.tex=.pdf)


%.pdf: %.tex
	pdflatex -shell-escape -interaction=nonstopmode "\input" $<
