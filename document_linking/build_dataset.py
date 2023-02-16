import os
import re

import json
import random
import sqlite3
import argparse

import pandas as pd
import numpy as np

from tqdm import tqdm
from functools import partial
from urllib.parse import unquote

import util

import multiprocessing as mp


def clean_url(
    url_string,
    regex_title=r"href=\"(.*?)\"&gt",
    ):
    try:
        n = re.search(regex_title, url_string)
        title_string = url_string[n.start()+6:n.end()-4]
        title_string = re.sub("%20", "_", title_string)
        title_string = re.sub("%23", "#", title_string)
        title_string = re.sub("%28", "(", title_string)
        title_string = re.sub("%29", ")", title_string)
        title_string = title_string.split("#")[0]
        # title_string = unquote(title_string)
        # title_string = title_string[0].upper() + title_string[1:]
        return title_string
    except:
        return False


def clean_text(
    text_string,
    regex_url=r"&lt;a(.*?)&lt;/a&gt;",
    regex_word=r"&gt;(.*?)&lt;"
    ):

    replace = {}
    for m in re.finditer(regex_url, text_string):
        start_idx = m.start()
        end_idx = m.end()

        url_title_string = text_string[start_idx:end_idx]
        n = re.search(regex_word, url_title_string)
        word_string = url_title_string[n.start()+4:n.end()-4]
        replace[url_title_string] = word_string

    for key, value in replace.items():
        text_string = re.sub(key, value, text_string)

    return text_string


# %run build_dataset.py \
#     --wiki_dump_path "/fsx/lintangsutawika/wikidump/enwiki/wiki/" \
#     --page_list_path "/fsx/lintangsutawika/page.tsv" \
#     --pagelinks_list_path "/fsx/lintangsutawika/pagelinks.tsv"

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki_dump_path", default=None, type=str)
    parser.add_argument("--lang", default=None, type=str)
    parser.add_argument("--page_list_path", default=None, type=str)
    parser.add_argument("--pagelinks_list_path", default=None, type=str)
    parser.add_argument("--language_list_path", default=None, type=str)
    parser.add_argument("--data_frame_path", default=None, type=str)
    parser.add_argument("--save_path", default="./", type=str)
    parser.add_argument("--samples", default=100, type=int)
    parser.add_argument("--num_proc", default=None, type=int)
    parser.add_argument("--connection", default="contigious", type=str)
    args = parser.parse_args()

    def get_article_file_location(file_path):

        article_file_path = {}
        article_file_idx = {}
        with open(file_path, "r") as file:
            for idx, line in enumerate(file.readlines()):
                article_dict = json.loads(line)
                id = int(article_dict["id"])

                article_file_path[id] = file_path
                article_file_idx[id] = int(idx)

        _df = page_df.copy()
        _df['file_path'] = _df['id'].map(article_file_path)
        _df['line'] = _df['id'].map(article_file_idx)
        return _df[_df['file_path'].notnull()]

    def sample_text(df, id_list=None, id_sample="sample", alpha=0.3):

        if id_list is not None:
            if id_sample == "sample":
                _df = df[df['id'].isin(id_list)]
            elif id_sample == "exclude":
                _df = df[~df['id'].isin(id_list)]
        else:
            _df = df

        sample_row = _df.sample(weights=_df.in_degree**alpha)
        article_file_path = sample_row['file_path'].values[0]
        article_file_idx = int(sample_row['line'].values[0])

        particular_line = linecache.getline(
            article_file_path,
            article_file_idx+1
            )
        line = json.loads(particular_line)
        sampled_text = clean_text(line['text'])
        inlink = sample_row['inlink'].values[0]

        sampled_text_list = re.split(
            r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', sampled_text
        )
        num_sentence = len(sampled_text_list)

        return sampled_text_list, num_sentence, inlink

    lang = args.lang
    wiki_dump_path = args.wiki_dump_path
    page_list_path = args.page_list_path
    pagelinks_list_path = args.pagelinks_list_path
    language_list_path = args.language_list_path
    data_frame_path = args.data_frame_path
    num_samples = args.samples
    num_proc = mp.cpu_count()-1 if args.num_proc == None else args.num_proc

    if lang is None:
        raise Exception("Need to define language")

    if args.data_frame_path is None:
        print("building new dataframe from data_frame_path....")
        if page_list_path is not None:
            print("Loading Pages")
            page_df = pd.read_csv(
                page_list_path, sep="\t",
                names=["id", "namespace", "title", "redirect", "_a", "_b", "_c"]
            )
            page_df = page_df[page_df['namespace'] == 0]
            page_df.drop(columns=["namespace", "_a", "_b", "_c"], inplace=True)
        else:
            raise Exception("Need to define page_list_path")

        if pagelinks_list_path is not None:
            print("Loading Page Links")
            pagelinks_df = pd.read_csv(
                pagelinks_list_path, sep="\t", engine="pyarrow",
                names=['id', 'namespace', 'title', 'from_namespace']
            )
            pagelinks_df = pagelinks_df[pagelinks_df['namespace'] == 0]
            pagelinks_df = pagelinks_df[pagelinks_df['from_namespace'] == 0]
            pagelinks_df.drop(columns=["namespace", "from_namespace"], inplace=True)
            pagelinks_df = pagelinks_df.groupby('title')['id'].apply(list)
            pagelinks_df = pagelinks_df.reset_index()
            pagelinks_df.rename(columns={"id": "inlink"}, inplace=True)
            pagelinks_df["in_degree"] = pagelinks_df["inlink"].apply(len)
        else:
            raise Exception("Need to define pagelinks_list_path")

        # Other languages that discuss the page, get id=x
        if language_list_path is not None:
            print("Loading Language Links")
            lang_df = pd.read_csv(
                language_list_path, sep="\t",
                names=["id", "language", "title"]
            )

        file_path = []
        for dir in sorted(os.listdir(wiki_dump_path)):
            subdir = os.path.join(wiki_dump_path, dir)
            file_list = sorted(os.listdir(subdir))
            for file in file_list:
                complete_file_path = os.path.join(subdir, file)
                file_path.append(complete_file_path)

        with mp.Pool(processes=num_proc) as pool:
            list_df = list(
                tqdm(
                    pool.imap(
                        get_article_file_location,
                        file_path
                    ),
                    total=len(file_path)
                    )
                )

        link_df = pd.concat(list_df, axis=0)
        link_df = pd.merge(link_df, pagelinks_df, on="title")
        link_df = link_df[link_df["redirect"] == 0]
        link_df = link_df[~link_df['title'].str.contains("(disambiguation)")]

        link_df.to_parquet(
            os.path.join(os.path.abspath(args.save_path), 'link_{}.parquet.gzip'.format(lang)),
            compression='gzip'
            )

    # %run build_dataset.py \
    #     --lang "en" \
    #     --data_frame_path "/fsx/lintangsutawika/augmented-pretraining/document_linking/link_en.parquet.gzip" \
    #     --connection "random"
    else:
        print("Loading {}".format(data_frame_path))
        link_df = pd.read_parquet(data_frame_path)

        import linecache

        collected_text = []
        for n in range(num_samples):

            num_sentence = 1
            while num_sentence == 1:
                sampled_text_list, num_sentence, inlink = sample_text(link_df)

            _idx = random.sample(range(1, num_sentence), 1)[0]
            segment_a = " ".join(sampled_text_list[:_idx])

            if args.connection == "contigious":
                verbalizers = [
                    "{} {}",
                    "{} is continued by {}",
                    "{} is followed by {}"
                ]

                segment_b = " ".join(sampled_text_list[_idx:])

            elif args.connection == "random":
                verbalizers = [
                    "{} has no connection to {}",
                    "{} is not connected to {}",
                    "{} is not linked to {}"
                ]

                alt_num_sentence = 1
                while alt_num_sentence == 1:
                    alt_sampled_text_list, alt_num_sentence, _ = sample_text(link_df, id_list=inlink, id_sample="exclude")

                _idx = random.sample(range(alt_num_sentence), 2)
                segment_b = " ".join(alt_sampled_text_list[min(_idx):max(_idx)])

            elif args.connection == "linked":
                verbalizers = [
                    "{} is linked to {}",
                    "{} has a connection to {}",
                    "{} is connected to {}"
                ]

                alt_num_sentence = 1
                while alt_num_sentence == 1:
                    alt_sampled_text_list, alt_num_sentence, _ = sample_text(link_df, id_list=inlink, id_sample="sample")

                _idx = random.sample(range(alt_num_sentence), 2)
                segment_b = " ".join(alt_sampled_text_list[min(_idx):max(_idx)])

            sampled_verbalizer = random.sample(verbalizers, 1)[0]
            text_line = sampled_verbalizer.format(segment_a, segment_b)
            collected_text.append(text_line)