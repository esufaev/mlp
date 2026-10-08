#!/bin/sh
# Сборка отчёта. Библиография оформлена через thebibliography, поэтому biber не нужен.
set -e
cd "$(dirname "$0")/tex"
PDFLATEX="pdflatex -interaction=nonstopmode -file-line-error"
$PDFLATEX 0-main > /dev/null
$PDFLATEX 0-main > /dev/null
$PDFLATEX 0-main | grep -E "^[^ ]+:[0-9]+:" || true
cp 0-main.pdf ../report.pdf
find . -maxdepth 1 -type f ! -name "*.tex" ! -name "*.cls" ! -name "*.sty" ! -name "*.clo" ! -name "*.sh" -delete
echo "report.pdf готов"
