import os
import re

import json
import random
import sqlite3
import argparse

import pandas as pd
import numpy as np

from tqdm import tqdm
from urllib.parse import unquote

parser = argparse.ArgumentParser()
parser.add_argument("--wiki_dump_path", default=None, type=str)
parser.add_argument("--page_list_path", default=None, type=str)
parser.add_argument("--redirect_list_path", default=None, type=str)
parser.add_argument("--data_frame_path", default=None, type=str)
parser.add_argument("--samples", default=10, type=int)
parser.add_argument("--num_proc", default=4, type=int)
args = parser.parse_args()

class WikiDump:
    """docstring for WikiDump"""
    def __init__(
        self,
        wiki_dump_path=None,
        page_list_path=None,
        redirect_list_path=None,
        data_frame_path=None,
        num_proc=4,
        regex_url=None,
        regex_title=None,
        regex_word=None,
    ):
        super(WikiDump, self).__init__()

        self.wiki_dump_path = wiki_dump_path
        self.page_list_path = page_list_path
        self.redirect_list_path = redirect_list_path
        self.data_frame_path = data_frame_path
        self.num_proc = num_proc

        if regex_url == None:
            self.regex_url = r"&lt;a(.*?)&lt;/a&gt;"
        else:
            self.regex_url = regex_url

        if regex_title == None:
            #self.regex_title = r"&gt;(.*?)&lt;"
            self.regex_title = r"href=\"(.*?)\"&gt"
        else:
            self.regex_title = regex_title

        if regex_word == None:
            self.regex_word = r"&gt;(.*?)&lt;"
        else:
            self.regex_word = regex_word

        if self.redirect_list_path is not None:
            self.redirect_df = pd.read_csv(
                self.redirect_list_path,
                sep="\t",
                names=["id", "namespace", "title"]
            )

            import ast
            def reformat(x):
                try:
                    if x[-1] != "\'":
                        x = x+"\'"

                    x = ast.literal_eval(
                        "b"+x
                    ).decode('utf-8')
                    x = re.sub("_", " ", x)

                except Exception as e:
                    print("Got error: {}".format(e))
                    print("On string: {}".format(x))

                return x

            self.redirect_df["title"] = self.redirect_df["title"].apply(reformat)

        if (self.data_frame_path is not None) and (os.path.exists(self.data_frame_path)):
            self.df = pd.read_pickle(self.data_frame_path)
        else:
            with open(self.page_list_path, "r") as file:
                self.df = pd.DataFrame(data={
                    "page": [re.sub("_", " ", page) for page in file.read().splitlines()[1:]],
                    })

            self.df["id"] = np.nan
            self.df["file_path"] = np.nan
            self.df["in_degree"] = 0
            self.df["linked"] = -1

            dir_list = os.listdir(self.wiki_dump_path)
            dir_list.sort()

            all_file_paths = []
            total_articles = 0
            for _dir in dir_list:
                subdir = os.path.join(self.wiki_dump_path, _dir)
                file_list = os.listdir(subdir)
                file_list.sort()

                article_id = {}
                article_text = {}
                article_file_path = {}
                for file in file_list:
                    all_file_paths.append(
                        os.path.join(subdir, file)
                    )

                    complete_file_path = os.path.join(subdir, file)

                    with open(complete_file_path, "r") as file:
                        for raw_file in file.readlines():
                            article_dict = json.loads(raw_file)

                            title = article_dict["title"]

                            # linked = self.get_linked_titles(article_dict["text"])
                            article_id[title] = article_dict["id"]
                            article_text[title] = article_dict["text"]
                            article_file_path[title] = complete_file_path
                            total_articles += 1

                article_list = list(article_file_path.keys())

                _df = self.df[self.df['page'].isin(article_list)]
                _df['file_path'] = _df['page'].map(article_file_path)
                _df['text'] = _df['page'].map(article_text)
                _df['id'] = _df['page'].map(article_id)

                def fn(x):
                    return self.get_linked_titles(x)

                _df['linked'] = _df['text'].apply(fn)

                self.df.loc[_df.index,:]=_df[:]

            # self.index_df = self.df[["page", "id"]]
            # self.index_df = self.index_df[self.index_df["id"].notnull()]
            # self.index_df.to_pickle("index_table.pkl")

            self.df = self.df[
                (self.df['file_path'].notnull())
                 # & (self.df['linked'] != -1)
            ]

            if type(self.data_frame_path) == str:
                self._process_link_count(self.data_frame_path)


    def _calc_in_degree(self, link_list, idx=None, save_path=None):
        for _i, link in tqdm(
            enumerate(link_list),
            total=len(link_list),
            desc=f"#{idx}",
            position=idx
        ):
            
            _idx = self.df[self.df['page'].isin(link)].index
            self.df.loc[_idx, "in_degree"] += 1

        if save_path != None:
            self.df[self.df['in_degree'] > 0].to_pickle(save_path)

    def _process_link_count(
        self,
        save_path
    ):
        from multiprocessing import Process, Manager

        def _split(list_object, num_split):
            k, m = divmod(len(list_object), num_split)
            return [list_object[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(num_split)]

        num_proc = self.num_proc
        if num_proc > 1:
            processes = []
            partitions = []
            # creating processes
            for idx, link_list in enumerate(_split(self.df['linked'][self.df['linked'] != -1].tolist(), num_proc)):

                _save_path = save_path.split(".")[0]+"_{}.pkl".format(idx)
                p = Process(
                    target=self._calc_in_degree,
                    args=(link_list, idx, _save_path,)
                )
                p.start()
                processes.append(p)
                partitions.append(_save_path)

            # completing process
            for p in processes:
                p.join()

            for part in tqdm(partitions):
                df = pd.read_pickle(part)
                idx = df[df['in_degree'] != 0].index
                self.df.loc[idx, "in_degree"] += df.loc[idx, "in_degree"]
                # os.remove(part)

        else:
            self.df["in_degree"] = 0
            for link in tqdm(self.df['linked'][self.df['linked'] != -1].tolist()):
                idx = self.df[self.df['page'].isin(link)].index
                self.df.loc[idx, "in_degree"] += 1

        self.df = self.df.sort_values('in_degree', ascending=False)
        self.df['in_degree'][self.df['in_degree'] == 0] = 1

        self.df.to_pickle(save_path)


    def _process_url_title_string(
        self,
        url_title_string
    ):

        try:
            n = re.search(self.regex_title, url_title_string)
            title_string = url_title_string[n.start()+6:n.end()-4]
            # title_string = re.sub("%20", " ", title_string)
            title_string = unquote(title_string)
            title_string = title_string[0].upper() + title_string[1:]
            return title_string
        except:
            return False
 

    def _clean_text(
        self,
        text
    ):

        replace = {}
        for m in re.finditer(self.regex_url, text):
            start_idx = m.start()
            end_idx = m.end()

            url_title_string = text[start_idx:end_idx]
            n = re.search(self.regex_word, url_title_string)
            word_string = url_title_string[n.start()+4:n.end()-4]
            replace[url_title_string] = word_string

        for key, value in replace.items():
            text = re.sub(key, value, text)

        return text


    def get_linked_titles(
        self,
        sampled_text
    ):

        connected_text = []
        for m in re.finditer(self.regex_url, sampled_text):
            start_idx = m.start()
            end_idx = m.end()

            url_title_string = sampled_text[start_idx:end_idx]
            title_string = self._process_url_title_string(url_title_string)

            if title_string != False:
                connected_text.append(title_string)

        if connected_text == []:
            return -1
        else:
            return connected_text


    def get_article_text(
        self,
        file_path,
        article,
        clean=True,
    ):
        
        article_list = {}
        with open(file_path, "r") as file:
            for line in file.readlines():
                _line = json.loads(line)
                _title = _line["title"]
                _text = _line["text"]

                if article == _title:
                    if clean == True:
                        _text = self._clean_text(_text)

                    return _text
        return -1


    def check_redirect(
        self,
        page_name,
    ):

        try:
            link_id = self.df[
                self.df["page"] == page_name
            ]['id'].values[0]
            redirect_page = self.redirect_df[
                self.redirect_df["id"] == int(link_id)
            ]["title"].values[0]
            # print("TRUE")
            # print("FROM {}".format(link))
            # print("TO {}".format(redirect_link))
            return redirect_page
        except:
            return page_name


    def get_sample(
        self,
        connection="contigous"
    ):

        assert connection in ["contigous", "random", "linked"]

        sampled_article = self.df[self.df['linked'] != -1].sample(1)
        sampled_page = sampled_article["page"].values[0]
        
        # sampled_page = self.check_redirect(sampled_page)
        # print(sampled_page)
        # sampled_article = self.df[self.df["page"] == sampled_page]
        sampled_file_path = sampled_article["file_path"].values[0]
        sampled_links = sampled_article["linked"].values[0]
        sampled_links = [re.sub("_", " ", link) for link in sampled_links]

        sampled_text = self.get_article_text(
            sampled_file_path, sampled_page, clean=True
        )

        if connection == "contigous":
            verbalizers = [
                "{} {}",
                "{} is continued by {}",
                "{} is followed by {}"
            ]

            sampled_text_list = re.split(
                r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', sampled_text
            )

            num_sentence = len(sampled_text_list)
            if num_sentence > 1:
                _idx = num_sentence//2
                segment_a = " ".join(sampled_text_list[:_idx])
                segment_b = " ".join(sampled_text_list[_idx:])
            else:
                segment_a = sampled_text
                segment_b = sampled_text
        else:

            segment_a = sampled_text
            if connection == "random":
                verbalizers = [
                    "{} has no connection to {}",
                    "{} is not connected to {}",
                    "{} is not linked to {}"
                ]

                not_sample_list = [sampled_page] + sampled_links
                random_article = self.df[
                    (self.df['linked'] != -1) & (~self.df["page"].isin(not_sample_list))
                    ].sample(1)
                random_page = random_article['page'].values[0]
                # random_page = self.check_redirect(random_page)
                # random_article = self.df[self.df["page"] == random_page]
                random_file_path = random_article['file_path'].values[0]

                segment_b = self.get_article_text(
                    random_file_path, random_page, clean=True
                )

            elif connection == "linked":
                verbalizers = [
                    "{} is linked to {}",
                    "{} has a connection to {}",
                    "{} is connected to {}"
                ]

                for idx, link in enumerate(sampled_links):
                    redirect_link = self.check_redirect(link)
                    if redirect_link != link:
                        sampled_links[idx] = redirect_link

                # Sampling which linked article to use
                # is proportional to the inverse of the number of in-degrees an article has
                # this is so that articles with high in-degrees are not over-sampled
                linked_article = self.df[
                    self.df["page"].isin(sampled_links)
                    ].sample(1,weights=1/self.df["in_degree"])

                linked_page = linked_article['page'].values[0]
                linked_file_path = linked_article['file_path'].values[0]
                linked = linked_article['linked'].values[0]

                segment_b = self.get_article_text(
                    linked_file_path, linked_page, clean=True
                )

        return segment_a, segment_b, random.sample(verbalizers, 1)[0]


if __name__ == '__main__':

    num_samples = args.samples
    wiki = WikiDump(
        wiki_dump_path=args.wiki_dump_path,
        page_list_path=args.page_list_path,
        redirect_list_path=args.redirect_list_path,
        data_frame_path=args.data_frame_path,
        num_proc=args.num_proc
    )

    # 1. Sample a text
    # 2. Parse through the URLs in the text
    # 3. Get term_to_search, which are the index and title string
    # 4. Get file from article_file_index
    # 5. Get from articles from get_text_from_dir
    # 6. Select the title string from the articles