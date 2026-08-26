# LRE manuscript

This directory contains the current manuscript source and compiled PDF in the
official Springer Nature `sn-jnl` journal-article format, together with its
bibliography, required local style dependencies, figures, and quantitative
claim map.

Build from this directory with:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The PDF is generated from the checked-in `main.tex`. Auxiliary build files are
ignored and are intentionally absent from the archive.
