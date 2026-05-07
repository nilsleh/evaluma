SPHINXBUILD = sphinx-build
SOURCEDIR   = docs
BUILDDIR    = docs/_build

.PHONY: html clean

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html

clean:
	rm -rf $(BUILDDIR)
