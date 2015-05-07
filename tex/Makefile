LATEX = pdflatex
#LATEX_FLAGS = -shell-escape -interaction=batchmode
LATEX_FLAGS = -shell-escape 

all: proposal.pdf proposal_slides.pdf slides.pdf assets/stagewise_model.pdf acl2015.pdf

gtm_cnn.pdf: gtm_cnn.py daft_init.py
	python $^

acl2015.pdf: acl2015.tex
	$(LATEX) $(LATEX_FLAGS) $^
	bibtex acl2015
	$(LATEX) $(LATEX_FLAGS) $^
	$(LATEX) $(LATEX_FLAGS) $^

proposal.pdf: proposal.tex
	$(LATEX) $(LATEX_FLAGS) $^
	biber proposal
	$(LATEX) $(LATEX_FLAGS) $^
	$(LATEX) $(LATEX_FLAGS) $^

proposal_slides.pdf: proposal_slides.tex gtm_cnn.pdf
	$(LATEX) $(LATEX_FLAGS) $^
	biber proposal_slides
	$(LATEX) $(LATEX_FLAGS) $^
	$(LATEX) $(LATEX_FLAGS) $^

slides.pdf: slides.tex gtm_cnn.pdf
	$(LATEX) $(LATEX_FLAGS) $^
	biber slides
	$(LATEX) $(LATEX_FLAGS) $^
	$(LATEX) $(LATEX_FLAGS) $^

.PHONY: clean

clean:
	rm *.pdf *.aux *.bbl *.bcf *.blg *.log *.nav *.run.xml *.snm *.toc
