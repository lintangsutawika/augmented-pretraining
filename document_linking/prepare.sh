#!/bin/bash

echo "dir: $1"
echo "lang: $2"
echo "timestamp: $3"

dir="$1"
lang="$2"
timestamp="$3"

ARTICLES="${lang}wiki-${timestamp}-pages-articles-multistream.xml.bz2"
TITLES="${lang}wiki-${timestamp}-all-titles-in-ns0.gz"
REDIRECT="${lang}wiki-${timestamp}-redirect.sql.gz"

for FILE in $ARTICLES $TITLES $REDIRECT
do
   if test -f "${dir}${FILE}"; then
      echo "$FILE exists."
   else
      echo "$FILE does not exists."
      wget "https://dumps.wikimedia.org/${lang}wiki/${timestamp}/"${FILE} -P "${dir}"
   fi
done

if test -f "${dir}/${lang}wiki-${timestamp}-all-titles-in-ns0"; then
   echo "${lang}wiki-${timestamp}-all-titles-in-ns0 exists."
else
   gunzip -d "${dir}/${lang}wiki-${timestamp}-all-titles-in-ns0.gz"
fi

if test -f "${dir}redirect.tsv"; then
   echo "redirect.tsv exists."
else
   python WikiUtils/parse_mysqldump.py \
      "${dir}/${lang}wiki-${timestamp}-redirect.sql.gz" \
      redirect \
      "${dir}/redirect.tsv"
fi

if test -f "${dir}wiki/"; then
   echo "wiki/ exists."
else
   python -m wikiextractor.WikiExtractor \
      "${dir}/${lang}wiki-${timestamp}-pages-articles-multistream.xml.bz2" \
      -o "${dir}/wiki/" \
      --json \
      --links
fi
