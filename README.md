# linked-document-pretraining
Generate Linked Document Pretraining data for LLMs

## Prerequisites

The files needed can be downloaed from [here](https://dumps.wikimedia.org/backup-index.html).

The preperation can be done with the script file.
```
sh prepare.sh <file directory> <language code> <date code>
```
`<lang code>` can be filled in with any language available on Wikipedia such as `en` for English. `<data code>` denotes the date the dump was taken. This is updated periodically so you might want the latest version available for example: `sh prepare.sh enwiki/ en 20220801`.

### Process Wikipedia Dump

Download the Wikipedia dump file.
```
<lang code>wiki-<date code>-pages-articles-multistream.xml.bz2
```
After downloading the file, we can extract it into something we can process later. We are using `--json` option to return the extracted version in json format with `--links` so that we can extract other pages that are mentioned within a page.
```
python -m wikiextractor.WikiExtractor <Wikipedia dump file> --json --links -o <save directory>
```

### Process Page Title List

Download the page title list file
```
<lang code>wiki-<date code>-all-titles-in-ns0.gz
```
which we can then decompress with `gunzip`. 

### Process Redirect List

Download the redirect list 
```
enwiki-20220801-redirect.sql.gz
```
and extract with WikiUtils.

## Data Preperation

### Build Dataset Generator

Before we can build the actual dataset, we will need to build the generator thet will sample from the Wikipedia pages, check for linked pages, and sample articles based on whether we want the the samples as `contigious`, `linked`, or `random`.

```
python build_dataset.py \
    --wiki_dump_path ../wikipedia/idwiki-text/ \
    --page_list_path idwiki/idwiki-20220801-all-titles-in-ns0 \
    --redirect_list_path idwiki/redirect_list.tsv \
    --data_frame_path idwiki_test.pkl \
    --num_proc 8
```

### Calculating In-Degrees for each Page

To prevent oversampling of highly cited pages in Wikipedia, we will need to calculate the in-degree value of each page. This is computationally demanding and is best done in a large multi-core setup in order to utilize multiprocessing features to break down the task. The number of cores used can be defined in the `num_proc` argument


## Sample Generation

TODO