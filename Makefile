.ONESHELL:
GCC_FLAGS = -Wall -fopenmp
GCC_L = -lOpenCL

ASSIGNMENT_GROUP=B
ASSIGNMENT_NUMBER=05
ASSIGNMENT_TITLE=pvs$(ASSIGNMENT_NUMBER)-grp$(ASSIGNMENT_GROUP)

COMPILE_TASK=0

.PHONY: build
build: matmult

.PHONY: debug
debug: GCC_FLAGS += -g
debug: build

.PHONY: matmult
matmult:
	g++ $(GCC_FLAGS) matmult.cpp -o matmult $(GCC_L) -DCOMPILE_TASK=$(COMPILE_TASK)

.PHONY: test
test: build
	./matmult

.PHONY: clean
clean:
	rm matmult

.PHONY: codeformat
codeformat:
	clang-format -i *.[ch]pp


PDF_FILENAME=$(ASSIGNMENT_TITLE).pdf
.PHONY: pdf
pdf:
	pandoc pvs.md -o $(PDF_FILENAME) --from markdown --template ~/.pandoc/eisvogel.latex --listings


FILES=Makefile README.md *.[ch]pp $(PDF_FILENAME)

ASSIGNMENT_DIR=$(ASSIGNMENT_TITLE)
TARBALL_NAME=$(ASSIGNMENT_TITLE)-piekarski-wichmann-ruckel.tar.gz
.PHONY: tarball
tarball: pdf
	[ -z "$(TARBALL_NAME)" ] || rm $(TARBALL_NAME)
	mkdir $(ASSIGNMENT_DIR)
	for f in $(FILES); do cp $$f $(ASSIGNMENT_DIR); done
	tar zcvf $(TARBALL_NAME) $(ASSIGNMENT_DIR)
	rm -fr $(ASSIGNMENT_DIR)
