# linked-document-pretraining
Generate Linked Document Pretraining data for LLMs

## To Generate Json files from the Wikipedia Dumps

`python -m wikiextractor.WikiExtractor <Wikipedia dump file> [--templates <extracted template file>]`

Articles Index `enwiki-20220801-pages-articles-multistream.xml.bz2`
Title Index `enwiki-20220801-all-titles-in-ns0.gz`
Redirect Index `enwiki-20220801-redirect.sql.gz`