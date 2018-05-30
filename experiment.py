#!/usr/bin/env python3

import argparse
import re
import random

from collections import defaultdict, Counter
from multiprocessing.dummy import Pool as ThreadPool

import analyze
import find_compounds


def read_decomp(fname):
    data = defaultdict(list)
    with open(fname) as fin:
        eng = ''
        for line in fin:
            line = line.strip()
            if line.startswith('#') and line[2] != '0' and line[2] != '1':
                match = re.match(r'\# .+ \{.*\} (\w+) (.+) (\w+) \+ (\w+) ; (.+) \+ (.+)', line)
                if match is not None:
                    lang = match.group(1)
                    whole = match.group(2)
                    left = match.group(3)
                    right = match.group(4)
                    eleft = match.group(5)
                    eright = match.group(6)
                    data[lang].append((eng, whole, left, right, eleft, eright))
            elif not line.startswith('#') and '||' not in line and ' words' not in line:
                eng = line.strip()
    return data


def random_test_set(decompfile):
    data = read_decomp(decompfile)

    test_set = []
    langs = random.sample(data.keys(), 100)
    for lang in langs:
        line = random.choice(data[lang])
        test_set.append((lang, line))
    return test_set


def read_testset(fname):
    data = []
    with open(fname) as f:
        for line in f:
            data.append(line.split('\t'))
    return data


def remove_from_dictionary(f2e, testset):
    for lang, _eng, word, *_ in testset:
        if word in f2e[lang]:
            del f2e[lang][word]
        else:
            print(word, 'not in dic')


def read_lr_counts(decomp):
    with open(decomp) as f:
        lr_counts = {}
        eng = ''
        left_counts = {}
        right_counts = {}

        for line in f:
            line = line.strip()
            if '||' in line:
                left, right = line.split('||')
                try:
                    lw, lc = left.strip().rsplit(' ', 1)
                    left_counts[lw.strip()] = int(lc)
                except:
                    pass
                try:
                    rw, rc = right.strip().rsplit(' ', 1)
                    right_counts[rw.strip()] = int(rc)
                except:
                    pass

            elif line == '':
                lr_counts[eng] = (dict(left_counts), dict(right_counts))
                left_counts.clear()
                right_counts.clear()

            elif line != '' and not line.startswith('#') and '||' not in line and ' words' not in line:
                eng = line
        lr_counts[eng] = (dict(left_counts), dict(right_counts))
        return lr_counts


def experiment_e2f():
    '''
    given an English word
    hyps = augment(english, lang)
    score!(hyps)
    is foreign word in hyps?
    '''
    testset = read_testset('testset')
    recipes = analyze.read_recipes('out/output/decomp-exp')
    f2e = find_compounds.load_dict('dictionaries/wiktionary.txt')
    remove_from_dictionary(f2e, testset)
    e2f = analyze.make_e2f_dict(f2e)
    # methods = ['concat', 'dropleft', 'glue', 'glue2']
    methods = ['concat', 'fuzzy']

    correct = 100
    for lang, eng, word, left, right, eleft, eright in testset:
        hyps = []
        for concept, newword in analyze.discover_compounds_single(
                lang, recipes, f2e, e2f, methods, english=eng, nodictcheck=True, flip=True):
            if newword[0] == word:  # word, lang, left, right, formation
                hyps.append(newword)

        if len(hyps) > 0:
            print(lang, hyps)
        else:
            correct -= 1
            print(lang, None)

    # todo: score hypotheses
    print(f'{correct} / 100')



def experiment_f2e():
    '''
    given a foreign word

    generate all using recipes
    if a hyp matches the word
        keep the english
        score the decomp
    rank by score

    '''
    testset = read_testset('testset')
    recipes = analyze.read_recipes('out/output/decomp-exp')
    leftrightcounts = read_lr_counts('out/output/decomp-exp')

    f2e = find_compounds.load_dict('dictionaries/wiktionary.txt')
    remove_from_dictionary(f2e, testset)
    e2f = analyze.make_e2f_dict(f2e)
    # methods = ['concat', 'dropleft', 'glue', 'glue2']
    methods = ['concat', 'glue']

    # print(leftrightcounts['hospital'])
    correct = 10
    # for lang, eng, word, left, right, eleft, eright in testset:

    def run(testword):
        lang = testword[0]
        eng = testword[1]
        word = testword[2]

        hyps = []
        for concept, newword in analyze.discover_compounds_single(
                lang, recipes, f2e, e2f, methods, nodictcheck=True, flip=True):
            if newword[0] == word:  # word, lang, left, right, formation
                hyps.append((concept, newword))

        # print(testword[0], testword[1], hyps)
        words = analyze.convert_wordlist_noeng([cw for concept, cw in hyps], f2e)

        result = []
        for (concept, hyp), w in zip(hyps, words):
            left, right = leftrightcounts[concept]
            w.update_score(left, right)
            result.append((round(w.score, 3), concept))

        # print(testword[0], result)

        return result

    pool = ThreadPool(10)
    results = pool.map(run, testset[90:100])

    for t, r in zip(testset[90:100], results):
        r = set(r)
        r = sorted(r, key=lambda x: -x[0])
        r = [(a[0], a[1]) for a in r]

        rank = -1
        for i, (score, eng) in enumerate(r):
            if eng == t[1]:
                rank = i
                break

        print(t, rank, r, sep='\t')

        if r == []:
            correct -= 1
    # for i, t in enumerate(testset):
    #     res = run(t)
    #     print(t[1], res)
    #     if res == []:
    #         correct -= 1

    print(f'{correct} / 100')

    # for tw, res in zip(testset, results):
    #     print(tw)
    #     print('\t', res)


all_langs = set()

def load_alltsv(file):
    e2f = defaultdict(list)
    f2e = defaultdict(list)
    with open(file) as f:
        for line in f:
            arr = line.strip('\n').split('\t')
            concept, lang, foreign, fleft, eleft, fright, eright, method = arr
            e2f[(concept, lang)].append((foreign, fleft, fright, eleft, eright, method))
            f2e[(foreign, lang)].append((concept, fleft, fright, eleft, eright, method))
            all_langs.add(lang)
    return e2f, f2e


def decompose_unk(f2e, lang, word):
    result = []
    for s1, s2 in find_compounds.split_word(word):
        if f2e[lang][s1] and f2e[lang][s2]:
            result.append((s1, s2))
    for s1, s2 in find_compounds.split_1glue(word):
        if f2e[lang][s1] and f2e[lang][s2]:
            result.append((s1, s2))
    return result


def demo():
    e2f_compounds, f2e_compounds = load_alltsv('demo/new-all.tsv')
    f2e = find_compounds.load_dict('dictionaries/wiktionary')

    e2f = analyze.make_e2f_dict(f2e)
    recipes = analyze.read_recipes('demo/new-decomp')
    leftrightcounts = read_lr_counts('demo/new-decomp')

    while True:
        try:
            target, lang, query = input('\n> ').split(' ', 2)

            if target == 'f':
                print('dictionary:')
                in_dict = False
                if lang == '*':
                    for ll in all_langs:
                        for result in e2f_compounds[(query, ll)]:
                            in_dict = True
                            print(result[0], result[1] + ' + ' + result[2], result[3] + ' + ' + result[4], result[5], sep=' || ')
                else:
                    for result in e2f_compounds[(query, lang)]:
                        in_dict = True
                        print(result[0], result[1] + ' + ' + result[2], result[3] + ' + ' + result[4], result[5], sep=' || ')

                # if not in_dict:
                print('\nhypotheses:')
                for hyp, score in generate_foreign_hyps(f2e, e2f, lang, recipes, query, leftrightcounts)[:20]:
                    print(hyp.orig, hyp.left + ' + ' + hyp.right, ','.join(hyp.leftengs) + ' + ' + ','.join(hyp.rightengs), score, sep=' || ')


            elif target == 'e':
                print('dictionary:')
                for result in f2e_compounds[(query, lang)]:
                    print(result[0], result[1] + ' + ' + result[2], result[3] + ' + ' + result[4], result[5], sep=' || ')

                print('\nsegmentation:')
                for left, right in decompose_unk(f2e, lang, query):
                    print(left + ' + ' + right, ','.join(f2e[lang][left]) + ' + ' + ','.join(f2e[lang][right]), sep=' || ')

        except Exception as e:
            print(e)
            continue

def load_forms():
    forms = defaultdict(lambda: Counter())
    with open('/home/wwu/worcomal/out/output/form') as f:
        for line in f:
            lang, form, count = line.split('\t')
            forms[lang][form] += int(count)

    for lang in forms:
        s = sum(forms[lang].values())
        for k in forms[lang]:
            forms[lang][k] /= s
    return forms


def generate_for_f2e(lang, expdir, unkwords):
    '''
    given a foreign word

    generate all using recipes
    if a hyp matches the word
        keep the english
        score the decomp
    rank by score

    '''
    recipes = analyze.read_recipes(expdir + '/decomp')
    leftrightcounts = read_lr_counts(expdir + '/decomp')

    f2e = find_compounds.load_dict('dictionaries/wiktionary.txt')
    e2f = analyze.make_e2f_dict(f2e)
    methods = ['concat', 'glue']

    # while True:
    #     lang, query = input('\n> ').split(' ', 1)
    #     if lang not in f2e:
    #         continue

    #     if query in f2e[lang]:
    #         print('in dictionary')
    #         for e in f2e[lang][query]:
    #             print(e)

    with open('/export/a08/wwu/worcomal-hyps/' + lang, 'w') as f:
        for concept, newword in analyze.discover_compounds_single(
                lang, recipes, f2e, e2f, methods, nodictcheck=True, flip=True):
            # newword = (word, lang, left, right, formation)

            # if newword[0] == query:
            print(concept, newword[0], newword[2], ','.join(f2e[lang][newword[2]]),
                newword[3], ','.join(f2e[lang][newword[3]]), newword[4],
                sep='\t', file=f)

    # def resolve(query):
    #     # resolved = defaultdict(list)

    #     for concept, newword in analyze.discover_compounds_single(
    #             lang, recipes, f2e, e2f, methods, nodictcheck=True, flip=True):
    #         # newword = (word, lang, left, right, formation)
    #         if newword[0] == query:
    #             print(concept, newword)
    #             # resolved[query].append((concept, query))

        # for word in resolved:
            # print(word, resolved[query])
    # print(testword[0], testword[1], hyps)
    # words = analyze.convert_wordlist_noeng([cw for concept, cw in hyps], f2e)

    # result = []
    # for (concept, hyp), w in zip(hyps, words):
    #     left, right = leftrightcounts[concept]
    #     w.update_score(left, right)
    #     result.append((round(w.score, 3), concept))


def generate_foreign_hyps(f2e, e2f, lang, recipes, concept, leftrightcounts):
    forms = load_forms()

    hyps = []
    for concept, newword in analyze.discover_compounds_single(
            lang, recipes, f2e, e2f, ['concat', 'glue'], english=concept, nodictcheck=True, flip=False):
            # for demo purposes, no flip
        hyps.append(newword)

    words = analyze.convert_wordlist_noeng(hyps, f2e)

    result = []
    for word in words:
        left, right = leftrightcounts[concept]
        word.update_score(left, right)
        # print(word.formation, forms[lang][word.formation])
        result.append((word, round(word.score + forms[lang][word.formation], 3)))

    result = list(set(result))
    return sorted(result, key=lambda x: x[1], reverse=True)


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('step')
    parser.add_argument('--decomp')

    return parser.parse_args()


def main():
    args = parseargs()

    if args.step == 'maketestset':
        test_set = random_test_set(args.decomp)
        for k, v in sorted(test_set):
            print(k + '\t' + '\t'.join(v))

    elif args.step == 'f2e':
        experiment_f2e()

    elif args.step == 'e2f':
        experiment_e2f()

    elif args.step == 'f2e-demo':
        generate_for_f2e('sw', 'demo', ['blah'])

    elif args.step == 'demo':
        demo()

    elif args.step == 'generate':
        generate_for_f2e('nb', 'demo', ['blah'])


if __name__ == '__main__':
    main()
