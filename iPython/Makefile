JUPYTER := jupyter
TIMEOUT := 3600

NOTEBOOKS = $(wildcard *.ipynb)
NOTEBOOKS_HTML = $(patsubst %.ipynb,out/%.html,$(NOTEBOOKS))

.PHONY: default notebooks_html clean

default: notebooks_html

out:
	mkdir out

notebooks_html: out $(NOTEBOOKS_HTML)

out/%.html: %.ipynb
	${JUPYTER} nbconvert --execute --ExecutePreprocessor.timeout=${TIMEOUT} --to html --log-level=20 --output-dir=out $<

clean:
	rm -rf out
