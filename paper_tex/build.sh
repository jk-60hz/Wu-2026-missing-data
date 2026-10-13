#!/usr/bin/env bash
# Build both papers and the arXiv upload bundles.
#   en: pdfLaTeX (arXiv default engine)      ko: XeLaTeX + kotex (Noto CJK KR fonts)
# Regenerate figures/tables first if results changed:
#   python scripts/make_report.py --paper && python scripts/make_tables_tex.py
set -euo pipefail
cd "$(dirname "$0")"
(cd en && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
(cd ko && latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex)

bundle() {  # $1 = en|ko
  local lang=$1 tmp
  tmp=$(mktemp -d)
  mkdir -p "$tmp/figures" "$tmp/tables"
  sed -e 's#{{../figures/}}#{{figures/}}#' -e "s#../tables/$lang#tables#" \
      -e 's#\\input{../figures/pipeline.tex}#\\input{figures/pipeline.tex}#' \
      -e 's#\\bibliography{../refs}#\\bibliography{refs}#' "$lang/main.tex" > "$tmp/main.tex"
  cp "$lang/main.bbl" "$tmp/main.bbl"
  cp refs.bib "$tmp/"
  cp figures/*.pdf figures/pipeline.tex "$tmp/figures/"
  cp tables/"$lang"/*.tex "$tmp/tables/"
  if [ "$lang" = ko ]; then
    printf 'This paper must be compiled with XeLaTeX (kotex, Noto CJK KR fonts).\n' > "$tmp/00README.txt"
  fi
  tar -czf "arxiv_$lang.tar.gz" -C "$tmp" .
  rm -rf "$tmp"
}
bundle en
bundle ko
echo "built: en/main.pdf ko/main.pdf arxiv_en.tar.gz arxiv_ko.tar.gz"
