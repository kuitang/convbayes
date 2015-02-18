LATEX = pdflatex
#LATEX_FLAGS = -shell-escape -interaction=batchmode
LATEX_FLAGS = -shell-escape 

all: proposal.pdf proposal_slides.pdf

proposal.pdf: proposal.tex
	$(LATEX) $(LATEX_FLAGS) $^
	biber proposal
	$(LATEX) $(LATEX_FLAGS) $^
	$(LATEX) $(LATEX_FLAGS) $^

proposal_slides.pdf: proposal_slides.tex
	$(LATEX) $(LATEX_FLAGS) $^
	biber proposal_slides
	$(LATEX) $(LATEX_FLAGS) $^
	$(LATEX) $(LATEX_FLAGS) $^

.PHONY: clean

clean:
	rm *.pdf *.aux *.bbl
