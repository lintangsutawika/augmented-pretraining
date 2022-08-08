import os
import re

import json
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--index", type=str)
parser.add_argument("--wiki_dir", type=str)
parser.add_argument("--samples", default=10, type=int)
args = parser.parse_args()

class WikiDump:
    """docstring for WikiDump"""
    def __init__(
        self,
        index_path,
        dump_path,
        regex_url=None,
        regex_title=None,
        regex_word=None,
    ):
        super(WikiDump, self).__init__()

        self.index_path = index_path
        self.dump_path = dump_path

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

        self.article_idx = {} #BidirectionalDict()
        with open(self.index_path, "r") as file:
            for line in file.readlines():
                file_index, article_index, *title = line[:-1].split(":")
                title = ":".join(title)
                article_index = int(article_index)
                # self.article_idx.__setitem__(title, article_index)
                self.article_idx[title] = article_index

        dir_list = os.listdir(self.dump_path)
        dir_list.sort()

        all_file_paths = []
        for _dir in dir_list:
            subdir = os.path.join(self.dump_path, _dir)
            file_list = os.listdir(subdir)
            file_list.sort()

            for file in file_list:
                all_file_paths.append(
                    os.path.join(subdir, file)
                )

        self.article_filename = {}
        for filename in all_file_paths:
            with open(filename, "r") as file:
                for text in file.readlines():
                    _text = json.loads(text)
                    _id = int(_text["id"])
                    _title = _text["title"]
                    self.article_filename[_title] = filename


    def _process_url_title_string(
        self,
        url_title_string
    ):
        n = re.search(self.regex_title, url_title_string)
        # title_string = url_title_string[n.start()+4:n.end()-4]
        title_string = url_title_string[n.start()+6:n.end()-4]
        title_string = re.sub("%20", " ", title_string)
        title_string = title_string[0].upper() + title_string[1:]
        return title_string


    def _get_clean_text(
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


    def get_article_from_file(
        self,
        file_path,
        article=None,
    ):
        
        article_list = {}
        with open(file_path, "r") as file:
            for line in file.readlines():
                _line = json.loads(line)
                _title = _line["title"]
                _text = _line["text"]

                if article is not None:
                    if article == _title:
                        article_list[_title] = _text
                        break
                else:
                    article_list[_title] = _text

        return article_list


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

            # if title_string in self.article_filename:
            if title_string in self.article_idx:
                file_path = self.article_filename[title_string]
                connected_text.append(title_string)

        return connected_text


    def get_text(
        self,
        article
    ):

        file_path = self.article_filename[article]
        text = self.get_article_from_file(file_path, article)[article]
        return text


    def get_sample(
        self
    ):

        article_titles = list(self.article_filename.keys())
        filtered_idx_list = []    
        linked_titles_list = []
        # while (sampled_sentence == ''):# 
        while (len(filtered_idx_list) == 0) and (len(linked_titles_list) == 0):
            # print(linked_titles_list)
            sampled_title = random.sample(article_titles, 1)[0]
            sampled_sentence = self.get_text(sampled_title)

            sampled_sentence_list = re.split(
                r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', sampled_sentence
            )

            for idx, sentence in enumerate(sampled_sentence_list[:-1]):
                linked_titles = self.get_linked_titles(sentence)
                if (re.search(self.regex_title, sentence)) and \
                    (len(linked_titles) != 0):
                    filtered_idx_list.append(idx)
                    linked_titles_list.append(linked_titles)

        _idx = random.randint(0, len(filtered_idx_list)-1)
        sampled_idx = filtered_idx_list[_idx]
        sampled_text = sampled_sentence_list[sampled_idx]
        sampled_link = linked_titles_list[_idx]
        
        contigous_sentence = " ".join(sampled_sentence_list[sampled_idx+1:])

        return {
            "sample": self._get_clean_text(sampled_text),
            "linked": sampled_link,
            "contigous": self._get_clean_text(contigous_sentence),
            }


if __name__ == '__main__':

    num_samples = args.samples
    wiki = WikiDump(
        index_path=args.index,
        dump_path=args.wiki_dir
        )

    # for i in range(num_samples):
    #     print("\n#####")
    #     sample = wiki.get_sample()
    #     sample
    #     print("#####\n")
        
    #Get Contigous
    #Get Linked
    #Get Random






    # 1. Sample a text
    # 2. Parse through the URLs in the text
    # 3. Get term_to_search, which are the index and title string
    # 4. Get file from article_file_index
    # 5. Get from articles from get_text_from_dir
    # 6. Select the title string from the articles