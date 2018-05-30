#!/usr/bin/env python3

from collections import defaultdict
import string
import unicodedata
from fuzzywuzzy import process


DEF_SEPARATOR = '###'


def normalize(text):
    return unicodedata.normalize('NFKC', text.casefold())


def load_david_panlex():
    import os
    dic = defaultdict(lambda: defaultdict(set))
    folder = '/export/a14/yarowsky-lab/lexicons/panlex'
    for f in os.listdir(folder):
        lang = f[0:3]
        with open(folder + '/' + f) as fin:
            for line in fin:
                arr = line.strip().split('\t')
                foreign = arr[0]
                english = arr[5]
                dic[lang][normalize(foreign)].add(normalize(english))
    return dic


def load_wiktionary():
    dic = defaultdict(lambda: defaultdict(set))

    with open('wiktionary_translations_all.csv') as fin:
        for line in fin:
            arr = line.strip().split('\t')
            if len(arr) < 4:
                continue
            lang = None
            english = ''
            if arr[2] == 'en':
                lang = arr[3]
                foreign = normalize(arr[1])
                english = normalize(arr[0])
            elif arr[3] == 'en':
                lang = arr[2]
                foreign = normalize(arr[0])
                english = normalize(arr[1])
            # todo: maybe have a prune_dictionary() function
            if lang is not None and lang[0].islower() and len(english) >= 1 \
                    and 'CJK' not in unicodedata.name(english[0]) and foreign not in string.ascii_lowercase:
                dic[lang][foreign].add(english)
    return dic


def load_wikt2dict():
    dic = defaultdict(lambda: defaultdict(set))
    with open('/export/a08/wwu/wikt2dict/dat/wiktionary/English/translation_pairs') as fin:
        for line in fin:
            arr = line.strip().split('\t')
            lang = None
            if arr[0] == 'en':
                lang = arr[2]
                foreign = normalize(arr[3])
                english = normalize(arr[1])
            if arr[2] == 'en':
                lang = arr[0]
                foreign = normalize(arr[1])
                english = normalize(arr[3])
            foreign = foreign.replace('[', '').replace(']', '')
            if lang is not None:
                dic[lang][foreign].add(english)
    return dic


def load_dict(filename):
    dic = defaultdict(lambda: defaultdict(set))
    with open(filename) as f:
        for line in f:
            lang, foreign, *english = line.strip().split('\t')
            dic[lang][foreign] = english
    return dic


def write_dict(dic, filename):
    with open(filename, 'w') as f:
        for lang in sorted(dic):
            for foreign in dic[lang]:
                print(lang, foreign, '\t'.join(dic[lang][foreign]), sep='\t', file=f)


def split_word(word):
    for i in range(1, len(word)):
        # allow multiword and hyphenated expressions
        yield word[:i].strip(' -'), word[i:].strip(' -')


def split_1glue(word):
    for i in range(1, len(word) - 1):
        yield word[:i], word[i + 1:]


def split_2glue(word):
    for i in range(1, len(word) - 2):
        yield word[:i], word[i + 2:]


def exact_match(dic, word):
    if word in dic:
        return word
    else:
        return None


def fuzzy_match(dic, word):
    result = process.extractOne(word, dic.keys())
    if result[1] >= 90:
        return result[0]
    else:
        return None


def find_match(lang_dic, word):
    entry = dict_match(lang_dic, word)
    if entry is None:
        entry = dict_match(lang_dic, word)
    return entry


def find_compounds(lang):
    lang_dic = dic[lang]
    print(lang)
    with open(output_folder + '/compounds.' + lang, 'w') as f:
        for word in lang_dic:
            for s1, s2 in split_word(word):
                # doesn't account for hyphen and space on foreign side
                # left = find_match(lang_dic, s1)
                # right = find_match(lang_dic, s2)
                # if left and right:
                #     print(word, left, right, sep='\t', file=f)

                entry1 = dict_match(lang_dic, s1)
                if entry1 is None:
                    entry1 = dict_match(lang_dic, s1 + '-')
                entry2 = dict_match(lang_dic, s2)
                if entry2 is None:
                    entry2 = dict_match(lang_dic, '-' + s2)
                if entry1 and entry2:
                    print(word, entry1, entry2, sep='\t', file=f)


def fuzzy_mid_dist(word1, word2):
    def mean(x, y):
        return (x + y) / 2
    def endness(i, word):  # currently middle is 0, make it 0.5
        # return abs(i - len(word) / 2) / (len(word) / 2)
        if i <= len(word) / 2:
            return (len(word) - i) / len(word)
        else:
            return i / len(word)

    table = [[0 for i in range(len(word2) + 1)] for i in range(len(word1) + 1)]
    for i in range(len(word1) + 1):
        table[i][0] = i
    for j in range(len(word2) + 1):
        table[0][j] = j
    for i in range(1, len(word1) + 1):
        for j in range(1, len(word2) + 1):
            factor = mean(endness(i, word1), endness(j, word2))
            table[i][j] = min(
                table[i - 1][j] + 1 * factor,
                table[i][j - 1] + 1 * factor,
                table[i - 1][j - 1] + (0 if word1[i - 1] == word2[j - 1] else 1) * factor
            )
    # for row in table:
    #     print(row)
    return table[len(word1)][len(word2)]


# def find_partition3(lang):
#     def partition3(word):
#         if len(word) <= 2:
#             return []
#         for a in range(0, len(word) - 1):
#             for b in range(a + 1, len(word)):
#                 left = word[:a]
#                 mid = word[a:b]
#                 right = word[b:]
#                 yield left, mid, right
#                 # print(left, mid, right)
#                 # if left in wordlist and mid in wordlist
#                 # print(left, '+', glue, '+', right)
#     lang_dic = dic[lang]
#     print(lang)
#     with open(output_folder + '/compounds.' + lang, 'w') as f:
#         for word in lang_dic:
#             for left, mid, right in partition3(word):
#                 if left

def check_bigrams():
    bigrams = set([])
    with open('bigrams') as f:
        for line in f:
            arr = line.strip().split('\t')
            bigrams.add(arr[1])

    dic = load_wiktionary()
    # for lang in dic:
    engs = dic['zh'].values()
    count = 0
    for bg in bigrams:
        if bg not in engs:
            count += 1
    total = len(bigrams)
    print(f'zh has {total - count} / {total}')


def parseargs():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('match', type=str, help='exact or fuzzy')
    parser.add_argument('output', type=str, help='output folder')
    parser.add_argument('--dict', type=str, choices=['wiktionary', 'panlex', 'wikt2dict'])
    return parser.parse_args()


if __name__ == '__main__':
    check_bigrams()
    asdf

    args = parseargs()
    dict_match = {'exact': exact_match, 'fuzzy': fuzzy_match}[args.match]

    load_dict_func = {
        'wiktionary': load_wiktionary,
        'panlex': load_david_panlex,
        'wikt2dict': load_wikt2dict
    }[args.dict]

    # global vars
    output_folder = args.output
    dic = load_dict_func()
    write_dict(dic, 'dictionaries/' + args.dict)

    from multiprocessing.dummy import Pool as ThreadPool
    pool = ThreadPool(10)
    langs = dic.keys()
    results = pool.map(find_compounds, langs)
