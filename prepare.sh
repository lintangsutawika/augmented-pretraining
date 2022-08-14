#!/bin/bash

echo "dir: $1"
echo "lang: $2"
echo "timestamp: $3"

dir="$1"
lang="$2"
timestamp="$3"

wget "https://dumps.wikimedia.org/${lang}wiki/${timestamp}/${lang}wiki-${timestamp}-pages-articles-multistream.xml.bz2" -P "${dir}"
wget "https://dumps.wikimedia.org/${lang}wiki/${timestamp}/${lang}wiki-${timestamp}-all-titles-in-ns0.gz" -P "${dir}"
wget "https://dumps.wikimedia.org/${lang}wiki/${timestamp}/${lang}wiki-${timestamp}-redirect.sql.gz" -P "${dir}"

gunzip -d "${dir}/${lang}wiki-${timestamp}-all-titles-in-ns0.gz" 

python WikiUtils/parse_mysqldump.py \
    "${dir}/${lang}wiki-${timestamp}-redirect.sql.gz" \
    redirect \
    "${dir}/redirect.tsv"

python -m wikiextractor.WikiExtractor \
    "${dir}/${lang}wiki-${timestamp}-pages-articles-multistream.xml.bz2" \
    -o "${dir}/wiki/" \
    --json \
    --links