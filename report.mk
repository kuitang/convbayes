JULIA_SRC  := $(shell find src -name '*.jl')
JULIA_HTML := $(JULIA_SRC:%.jl=html/%.html)

# Handy variable reference (to implement Don't Repeat Yourself)
# $@ -- name of the target
# $< -- name of the first prerequisite
# $^ -- name of all prerequisites, separated by spaces

all: $(JULIA_HTML)

# TODO: Make jocco specify the output file, be smarter about stuff.
html/src/%.html: src/%.jl
	@echo "report.mk making $@: $^"
	. env.sh ; julia jocco.jl $^
	
