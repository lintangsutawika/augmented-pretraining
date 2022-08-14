# linked-document-pretraining
Generate Linked Document Pretraining data for LLMs

## To Generate Json files from the Wikipedia Dumps

1. Preprocessing the Wikipedia dump.

Download the latest articles index file `enwiki-20220801-pages-articles-multistream.xml.bz2`

`python -m wikiextractor.WikiExtractor <Wikipedia dump file> --json --links`

2. Preprocessing the list of page titles

Download Title index `enwiki-20220801-all-titles-in-ns0.gz` and decompress with `gunzip`

3. Preprocess the redirect list

Download the list `enwiki-20220801-redirect.sql.gz` and extract with WikiUtils


## Calculating In-Degrees for each Page

To prevent oversampling of highly cited pages in Wikipedia, we will need to calculate the in-degree value of each page. This is computationally demanding and is best done in a large multi-core setup in order to utilize multiprocessing features to break down the task. The number of cores used can be defined in the `num_proc` argument

```
sh prepare.sh idwiki/ id 20220801
```

## DataFrame Preperation

```
python build_dataset.py \
    --wiki_dump_path ../wikipedia/idwiki-text/ \
    --page_list_path idwiki/idwiki-20220801-all-titles-in-ns0 \
    --redirect_list_path idwiki/redirect_list.tsv \
    --data_frame_path idwiki_test.pkl \
    --num_proc 8
```

## Sample Generation