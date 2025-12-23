# Makefile for Sphinx documentation
#
# Usage:
#   make html     - Build HTML documentation
#   make clean    - Remove build artifacts
#   make linkcheck - Check external links
#   make doctest  - Run doctests in documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  html      - Build HTML documentation"
	@echo "  clean     - Remove build artifacts"
	@echo "  linkcheck - Check external links"
	@echo "  doctest   - Run doctests in documentation"
	@echo "  livehtml  - Start live-reload server (requires sphinx-autobuild)"

.PHONY: html
html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
	@echo
	@echo "Build finished. Open $(BUILDDIR)/html/index.html to view."

.PHONY: clean
clean:
	rm -rf "$(BUILDDIR)"

.PHONY: linkcheck
linkcheck:
	$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS)

.PHONY: doctest
doctest:
	$(SPHINXBUILD) -b doctest "$(SOURCEDIR)" "$(BUILDDIR)/doctest" $(SPHINXOPTS)

.PHONY: livehtml
livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
