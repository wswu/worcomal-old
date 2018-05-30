#!/usr/bin/env python3

import copy
import math
import os
import string
import sys
import time
from collections import Counter, defaultdict, namedtuple
from functools import lru_cache
from itertools import groupby, product
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
from sklearn import cluster
from scipy.cluster.hierarchy import fcluster, linkage

import editdistance
import find_compounds
import nltk
import processes
from nltk.corpus import wordnet as wn
from nltk.tag.stanford import StanfordPOSTagger
from tqdm import tqdm
from unidecode import unidecode
import dwhash

from fuzzywuzzy import process

class CompoundWord(object):
    def __init__(self, eng, lang, orig, left, leftengs, right, rightengs, formation):
        self.eng = eng
        self.lang = lang
        self.orig = orig
        self.left = left
        self.leftengs = leftengs
        self.right = right
        self.rightengs = rightengs
        self.formation = formation

        # set later
        self.fliporder = False
        self.score = None
        self.bestleft = None
        self.bestright = None

    def update_score(self, left_counts, right_counts):
        '''Update self.score, self,bestleft, and self.bestright'''

        if len(self.leftengs) == 0 or len(self.rightengs) == 0:
            return

        inorder_left = max(self.leftengs, key=lambda x: left_counts.get(x, 0))
        inorder_right = max(self.rightengs, key=lambda x: right_counts.get(x, 0))
        inorder_score = math.sqrt(left_counts.get(inorder_left, 0.1) * right_counts.get(inorder_right, 0.1))

        flipped_left = max(self.leftengs, key=lambda x: right_counts.get(x, 0))
        flipped_right = max(self.rightengs, key=lambda x: left_counts.get(x, 0))
        flipped_score = math.sqrt(right_counts.get(flipped_left, 0.1) * left_counts.get(flipped_right, 0.1))

        if inorder_score >= flipped_score:
            self.score = inorder_score
            self.bestleft = inorder_left
            self.bestright = inorder_right
        else:
            self.score = flipped_score
            self.bestleft = flipped_left
            self.bestright = flipped_right

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(self, other.__class__):
            return self.lang == other.lang and self.orig == other.orig and self.left == other.left and self.right == other.right
        return False

    def __hash__(self):
        return hash((self.lang, self.orig, self.left, self.right))


def load_wordlists(folder, formation=None):
    result = []
    for f in os.listdir(folder):
        lang = f.split('.')[1]
        with open(f'{folder}/{f}') as fin:
            for line in fin:
                arr = line.strip().split('\t')
                if len(arr) > 3:
                    print('bad:', arr)
                else:
                    orig, left, right = arr  # line.strip().split('\t')
                    result.append((orig, lang, left, right, formation))
    return result


def load_aug_wordlists(folder):
    result = []
    for f in os.listdir(folder):
        lang = f.split('.')[1]
        with open(f'{folder}/{f}') as fin:
            for line in fin:
                arr = line.strip().split('\t')
                result.append(arr)  # orig, lang, left, right, formation
    return result


def load_words_tsv(file):
    result = []
    with open(file) as f:
        for line in f:
            arr = line.strip().split('\t')
            arr[4] = tuple(arr[4].split(','))  # leftengs
            arr[6] = tuple(arr[6].split(','))  # rightengs
            result.append(CompoundWord(*arr))
    return result


def convert_wordlist(wordlist, f2e):
    result = []
    for orig, lang, left, right, formation in wordlist:
        engs = f2e[lang][orig]
        leftengs = tuple(f2e[lang][left])
        rightengs = tuple(f2e[lang][right])
        for eng in engs:
            word = CompoundWord(eng, lang, orig, left, leftengs, right, rightengs, formation)
            result.append(word)
    return result


def convert_wordlist_noeng(wordlist, f2e):
    result = []
    for orig, lang, left, right, formation in wordlist:
        leftengs = tuple(f2e[lang][left])
        rightengs = tuple(f2e[lang][right])

        word = CompoundWord(None, lang, orig, left, leftengs, right, rightengs, formation)
        result.append(word)
    return result


def write_single_file(words, outputfile):
    with open(outputfile, 'w') as fout:
        for word in sorted(words, key=lambda x: (x.eng, x.lang)):
            print(word.eng, word.lang, word.orig, word.left, ','.join(word.leftengs),
                  word.right, ','.join(word.rightengs), word.formation, sep='\t', file=fout)


def languages_that_compound(folder):
    line_counts = []
    for f in os.listdir(folder):
        with open(folder + '/' + f) as fin:
            lang = f[f.index('.') + 1:]
            line_counts.append((lang, sum(1 for line in fin)))
    line_counts.sort(key=lambda x: -x[1])
    return line_counts


def get_left_right_counts(data):
    left = Counter()
    right = Counter()
    for entry in data:
        for s in entry.left_eng.split(','):
            left[s] += 1
        for s in entry.right_eng.split(','):
            right[s] += 1
    return left, right


def normalize(counter):
    fsum = 0
    for word, count in counter.items():
        fsum += count
    for word in counter:
        counter[word] = counter[word] * 1.0 / fsum


def maxcount(counter, items):
    return max([counter.get(x, 0) for x in items]) if len(items) > 0 else 0


def reorder_components(words, left, right):
    inorder = []
    backwards = []
    for word in words:
        best_inorder = maxcount(left, word.leftengs) + maxcount(right, word.rightengs)
        best_backwards = maxcount(left, word.rightengs) + maxcount(right, word.leftengs)
        if best_inorder > best_backwards:
            inorder.append(word)
        else:
            backwards.append(word)
    if len(inorder) < len(backwards):
        return right, left
    else:
        return left, right


def pprint(component_counts, outputfile, max_comps=5):
    left, right = component_counts
    left_sorted = Counter(left).most_common(max_comps)
    right_sorted = Counter(right).most_common(max_comps)

    leftmax = max(len(comp) for comp, count in left_sorted)
    rightmax = max(len(comp) for comp, count in right_sorted)

    for i in range(max(len(left_sorted), len(right_sorted))):
        if i < len(left_sorted):
            print(f"{left_sorted[i][0]:{leftmax}s} {left_sorted[i][1]:2d} || ", end='', file=outputfile)
        else:
            print(' ' * (leftmax + 4) + "|| ", end='', file=outputfile)
        if i < len(right_sorted):
            print(f"{right_sorted[i][0]:{rightmax}s} {right_sorted[i][1]:2d}  ", file=outputfile)
        else:
            print(' ' * (rightmax + 5), file=outputfile)


def compute_component_distances(words):
    dist = defaultdict(lambda: 0.5)
    for word in words:
        for l1 in word.leftengs:
            for l2 in word.leftengs:
                dist[l1, l2] = 0
                dist[l2, l1] = 0
        for r1 in word.rightengs:
            for r2 in word.rightengs:
                dist[r1, r2] = 0
                dist[r2, r1] = 0
        for l in word.leftengs:
            for r in word.rightengs:
                dist[l, r] = 1
                dist[r, l] = 1
    return dist


def flatten(seq):
    return [x for sub in seq for x in sub]

def new_cluster(words):
    words = filter_comp_engs(words, at_least=2)
    dist = compute_component_distances(words)
    components = sorted(set(flatten([w.leftengs + w.rightengs for w in words])))

    if len(components) < 2:
        return {}, {}

    def dist_func(pair):
        i, j = pair
        return dist[components[i], components[j]]

    triu = np.triu_indices(len(components), 1)
    distances = np.apply_along_axis(dist_func, 0, triu)
    linkage_matrix = linkage(distances, method='ward')
    clusters = fcluster(linkage_matrix, 2, criterion='maxclust')

    newleft = []
    newright = []
    for comp, clus in zip(components, clusters):
        if clus == 1:
            newleft.append(comp)
        else:
            newright.append(comp)
    return newleft, newright


def group_by_english(words):
    result = defaultdict(list)
    for word in words:
        result[word.eng].append(word)
    return result


def filter_comp_engs(words, at_least):
    counts = Counter()
    for w in words:
        for e in w.leftengs:
            counts[e] += 1
        for e in w.rightengs:
            counts[e] += 1

    filtered_words = []
    for w in words:
        leftengs = [e for e in w.leftengs if counts[e] >= at_least and e != '']
        rightengs = [e for e in w.rightengs if counts[e] >= at_least and e != '']
        newword = copy.deepcopy(w)
        newword.leftengs = leftengs
        newword.rightengs = rightengs
        filtered_words.append(newword)
    return filtered_words


def english_to_components_nocluster(words, at_least=2):
    words = filter_comp_engs(words, at_least)
    result = defaultdict(lambda: (Counter(), Counter()))
    for word in words:
        for e in word.leftengs:
            result[word.eng][0][e] += 1
        for e in word.rightengs:
            result[word.eng][1][e] += 1
    return result


def english_to_components(words):
    global num_threads
    pool = ThreadPool(num_threads)
    grouped = list(group_by_english(words).items())
    results = pool.map(new_cluster, [v for k, v in grouped])
    return {g[0]: left_right for g, left_right in zip(grouped, results)}


def save_decomp_file(words, eng2comps, e2f, output_file):
    grouped = group_by_english(words)
    with open(output_file, 'w') as f:
        for eng, (left_cluster, right_cluster) in sorted(eng2comps.items()):
            left_cluster, right_cluster = eng2comps[eng]

            # count English definitions of components, but only one per language
            origleft = Counter()
            done = set()
            for w in grouped[eng]:
                for compeng in w.leftengs:
                    if (compeng, w.lang) not in done:
                        origleft[compeng] += 1
                        done.add((compeng, w.lang))

            origright = Counter()
            done = set()
            for w in grouped[eng]:
                for compeng in w.rightengs:
                    if (compeng, w.lang) not in done:
                        origright[compeng] += 1
                        done.add((compeng, w.lang))

            newleft = {x: origleft.get(x, 0) + origright.get(x, 0) for x in left_cluster}
            newright = {x: origleft.get(x, 0) + origright.get(x, 0) for x in right_cluster}
            left, right = reorder_components(grouped[eng], newleft, newright)

            left = {k: v for k, v in left.items() if v >= 2}
            right = {k: v for k, v in right.items() if v >= 2}

            if len(left) > 0 and len(right) > 0:
                for w in grouped[eng]:
                    w.update_score(left, right)
                print(eng, file=f)
                pprint([left, right], outputfile=f, max_comps=10)
                for word in sorted(grouped[eng], key=lambda x: -x.score):
                    trans = transliterations(word, e2f)
                    print('#', f'{word.score:.3f}', trans,
                        word.lang, word.orig, word.left, '+', word.right, ';', word.bestleft, '+', word.bestright, file=f)
                print(len(grouped[eng]), 'words', file=f)
                print(file=f)


InfluentialLangs = processes.influential_languages_map()

def transliterations(word, e2f):
    dists = defaultdict(list)
    for lang in InfluentialLangs[word.lang]:
        for influencer in e2f[word.eng][lang]:
            if influencer != word.orig:
                d = editdistance.eval(unidecode(word.orig), unidecode(influencer))
                if d < 0.5 * len(word.orig):  # todo: filter later instead of here
                    dists[d].append((influencer, lang))
    d = editdistance.eval(unidecode(word.orig), word.eng)
    if d < 0.5 * len(word.orig):
        dists[d].append((word.eng, 'en'))
    return dict(dists)


def read_recipes(file):
    with open(file) as f:
        recipes = {}
        lefts = []
        rights = []
        f.readline()  # skip first line
        eng = ''
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            elif line == '':  # last line of file is also blank
                recipes[eng] = (tuple(lefts), tuple(rights))
                lefts.clear()
                rights.clear()
            elif '||' in line:
                left, right = line.split('||')
                left = left.strip().rsplit(' ', 1)[0].strip()
                right = right.strip().rsplit(' ', 1)[0].strip()
                if left != '':
                    lefts.append(left)
                if right != '':
                    rights.append(right)
            elif ' words' not in line:
                eng = line
        return recipes


@lru_cache(maxsize=None)
def _wn_dist(tup):
    x, y = tup
    xsyns = wn.synsets(x)
    ysyns = wn.synsets(y)
    if len(xsyns) == 0 or len(ysyns) == 0:
        return 0.0
    return 1 - max(wn.wup_similarity(xs, ys) or 0 for xs in xsyns for ys in ysyns)


def wn_dist(word1, word2):
    return _wn_dist(tuple(sorted([word1, word2])))


_path_to_model = '/home/wwu/softw/stanford-postagger-2017-06-09/models/english-bidirectional-distsim.tagger'
_path_to_jar = '/home/wwu/softw/stanford-postagger-2017-06-09/stanford-postagger.jar'
stanford = StanfordPOSTagger(_path_to_model, _path_to_jar)


def load_tags():
    dic = {}
    with open('tags-redo') as f:
        for line in f:
            word, tag = line.strip().split('\t')
            dic[word] = tag
    return dic


tagdict = load_tags()


def postag(word, redo=False):
    if word not in tagdict or redo:
        tag = stanford.tag([word])[-1][1][0]
        # -1 for last word (head of multiword phrase), 1 for pos tag, 0 for first letter only
        tagdict[word] = tag
    return tagdict[word]
    # return nltk.pos_tag([word])[0][1]


def load_glove(path, dim):
    glove = defaultdict(lambda: [0.0 for _ in range(dim)])
    with open(path) as fin:
        for line in fin:
            word, *vec = line.strip().split(' ')
            vec = [float(x) for x in vec]
            glove[word] = vec
    return glove


GloveVectors = None


def cosine_dist(word1, word2):
    global GloveVectors
    if GloveVectors is None:
        # print('Loading glove vectors')
        sys.stderr.write('loading glove vectors\n')
        GloveVectors = load_glove('/export/a08/wwu/res/glove/glove.6B.100d.txt', 100)
    word1 = word1.lower()
    word2 = word2.lower()
    # vec1 = GloveVectors.get(word1, np.mean([GloveVectors[x] for x in word1.split()], axis=0))
    # vec2 = GloveVectors.get(word2, np.mean([GloveVectors[x] for x in word2.split()], axis=0))
    vec1 = GloveVectors[word1]
    vec2 = GloveVectors[word2]
    result = np.inner(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return 1 - (result if not np.isnan(result) else 0)


def discover_fuzzy_mid(recipes, f2e, e2f, outputdir):
    def discover(lang):
        print(lang)
        with open(f'{outputdir}/new.{lang}', 'w') as f:
            for word in tqdm(f2e[lang]):
                for lefts, rights in recipes:
                    for eleft, eright in product(lefts, rights):
                        for fleft, fright in product(e2f[eleft][lang], e2f[eright][lang]):
                            newword = fleft + fright
                            if word[0] != newword[0] and word[-1] != newword[-1]:
                                continue
                            dist = find_compounds.fuzzy_mid_dist(newword, word)
                            if dist < len(word) * 0.25:
                                found = (newword, lang, fleft, fright, 'fuzzy mid')
                                print('\t'.join(found), file=f)
    global num_threads
    pool = ThreadPool(num_threads)
    pool.map(discover, ['it'])  # sorted(f2e.keys()))


CyrillicLowercase = set('абвгдежзийклмнопрстуфхцчшщьюя')

def discover_new_compounds(recipes, f2e, e2f, outputdir, methods, singlelang=None):
    def discover(lang):
        with open(f'{outputdir}/new2.{lang}', 'w') as f:  # TODO: don't hardcode
            start = time.perf_counter()
            words = discover_compounds_single(lang, recipes, f2e, e2f, methods)
            for concept, word in words:
                print('\t'.join(word), file=f)
            elapsed = time.perf_counter() - start
            print(lang, elapsed)

    if singlelang is not None:
        print('single lang is', singlelang)
        discover(singlelang)
    else:
        print('threading?')
        global num_threads
        pool = ThreadPool(num_threads)
        pool.map(discover, sorted(f2e.keys(), reverse=True))



def discover_compounds_single(lang, recipes, f2e, e2f, methods, english=None, nodictcheck=False, flip=False):
    lsh = None
    # if 'fuzzy' in methods:
    #     lsh = dwhash.load(lang)

    newwords = set()
    if english is not None:
        recipes = {english: recipes[english]}

    for concept, (lefts, rights) in recipes.items():
        for eleft, eright in product(lefts, rights):
            for fleft, fright in product(e2f[eleft][lang], e2f[eright][lang]):
                if fleft[0] in CyrillicLowercase:
                    alphabet = list(CyrillicLowercase)
                else:
                    alphabet = list(string.ascii_lowercase)
                # alphabet = []  # TODO REMOVE
                alphabet.extend(['-', ' '])

                for i in [0, 1] if flip else [0]:
                    # do it again with flipped order
                    if i == 1:
                        fleft, fright = fright, fleft

                    # concat
                    if 'concat' in methods:
                        newword = fleft + fright
                        found = (newword, lang, fleft, fright, 'concat')
                        if (nodictcheck or newword in f2e[lang]) and found not in newwords:
                            # newwords.add(found)
                            yield concept, found

                    # glue char
                    if 'glue' in methods:
                        for glue in alphabet:
                            newword = fleft + glue + fright
                            found = (newword, lang, fleft, fright, 'glue ' + glue)
                            if (nodictcheck or newword in f2e[lang]) and found not in newwords:
                                # newwords.add(found)
                                yield concept, found

                    # drop one off left
                    if 'dropleft' in methods:
                        newword = fleft[:-1] + fright
                        found = (newword, lang, fleft, fright, 'drop left')
                        if (nodictcheck or newword in f2e[lang]) and len(fleft) >= 2 and found not in newwords:
                            # newwords.add(found)
                            yield concept, found

                    # 2 glue chars
                    if 'glue2' in methods:
                        for g1 in alphabet:
                            for g2 in alphabet:
                                newword = fleft + g1 + g2 + fright
                                found = (newword, lang, fleft, fright, 'glue ' + g1 + g2)
                                if (nodictcheck or newword in f2e[lang]) and found not in newwords:
                                    # newwords.add(found)
                                    yield concept, found

                    # fuzzy
                    if 'fuzzy' in methods:
                        newword = fleft + fright
                        found = (newword, lang, fleft, fright, 'fuzzy')
                        # if (nodictcheck or newword in f2e[lang]) and found not in newwords:
                        possible = process.extract(newword, e2f[concept][lang])

                        # for word in dwhash.lookup(lsh, newword):
                        for word, score in possible:
                            if score >= 70 and found not in newwords:
                                yield concept, found


def analyze_word_formation(words, outputfile):
    result = defaultdict(Counter)
    for word in words:
        result[word.lang][word.formation] += 1
    with open(outputfile, 'w') as f:
        for lang in sorted(result):
            for form in sorted(result[lang]):
                print(lang, form, result[lang][form], sep='\t', file=f)


def analyze_word_formation(words, outputfile):
    result = defaultdict(Counter)
    glues = defaultdict(Counter)
    for word in words:
        result[word.lang][word.formation.split(' ')[0]] += 1
        if word.formation.split(' ')[0] == 'glue':
            glues[word.lang][word.formation.split(' ', 1)[1]] += 1

    with open(outputfile, 'w') as f:
        for lang in sorted(result):
            normalize(result[lang])

            print(lang,
                round(result[lang]['concat'], 2),
                round(result[lang]['drop'], 2),
                round(result[lang]['glue'], 2),
                ', '.join([x for x, c in glues[lang].most_common(2)]),
                sep=' & ', end=' \\\\\n', file=f)
            # for form in sorted(result[lang]):
                # print(lang, form, round(result[lang][form], 2), sep='\t', file=f)



def make_e2f_dict(f2e):
    e2f = defaultdict(lambda: defaultdict(set))
    for lang in f2e:
        for f in f2e[lang]:
            for e in f2e[lang][f]:
                e2f[e][lang].add(f)
    return e2f


def augment(words, f2e, compsfile, outputname, forms):
    recipes = read_recipes(compsfile)
    e2f = make_e2f_dict(f2e)

    os.makedirs(outputname, exist_ok=True)
    discover_new_compounds(recipes, f2e, e2f, outputname, forms)
    # discover_fuzzy_mid(recipes, f2e, e2f, outputname)

    # cat new/* > new.txt
    with open(outputname + '-found', 'w') as f:
        for n in load_aug_wordlists(outputname):
            print('\t'.join(n), file=f)

    newwords = []
    with open(outputname + '-found') as fin:
        for line in fin:
            arr = line.strip('\n').split('\t')
            newwords.append(arr)

    print('loaded new words', len(newwords))
    for w in convert_wordlist(newwords, f2e):
        if w not in words:
            words.add(w)
    print('total words', len(words))

    save_decomp_file(words, english_to_components(words), e2f, outputname + '-decomp')
    write_single_file(words, outputname + '-all.tsv')


num_threads = 1

def parseargs():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', nargs='+',
        help='group,decomp,augment')
    parser.add_argument('-d', '--dict',
        help='location of f2e dictionary')
    parser.add_argument('-i', '--input',
        help='for genoutput or interactive, location of final.tsv')
    parser.add_argument('-o', '--output',
        help='output folder')
    parser.add_argument('-s', '--sim',
        help='similarity metric for clustering',
        choices=['wn', 'glove'],
        default='wn')
    parser.add_argument('--forms', nargs='*', default=['glue', 'dropleft'])
    parser.add_argument('--single')
    parser.add_argument('--singlelang')
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--testset',
        help='if there is a testset, remove these words from the dictionary')

    parser.add_argument('--adddata',
        help='a compoound list generated by find_compounds.py')
    return parser.parse_args()


def main():
    args = parseargs()
    global num_threads
    num_threads = args.num_threads

    print('loading dict')
    f2e = find_compounds.load_dict(args.dict)

    print('loading words')
    if 'group' in args.steps:
        words = convert_wordlist(load_wordlists(args.input, formation='concat'), f2e)
        # filter out affixes?
        # words = [w for w in words if w.orig[0] != '-']
        write_single_file(words, args.output + '/all.tsv')
    else:
        words = load_words_tsv(args.input)
    words = set(words)

    if args.single is not None:
        words = [w for w in words if w.eng == args.single]
    print('loaded words', len(words))

    if 'decomp' in args.steps:
        if args.testset:
            import experiment
            testset = experiment.read_testset(args.testset)
            experiment.remove_from_dictionary(f2e, testset)
        e2f = make_e2f_dict(f2e)
        save_decomp_file(words, english_to_components(words), e2f, args.output + '/decomp')

    if args.adddata:
        newwords = convert_wordlist(load_wordlist(args.adddata, formation='concat'), f2e)
        words.extend(newwords)
        save_decomp_file(words, english_to_components(words), e2f, args.output + '/')

    if 'augment' in args.steps:
        augment(words, f2e, args.output + '/decomp', args.output + '/new', ['glue', 'dropleft'])

    if 'augmentfuzzy' in args.steps:
        augment(words, f2e, args.output + '/decomp', args.output + '/fuzzy', ['fuzzy'])

    if 'augment2' in args.steps:
        augment(words, f2e, args.output + '/new-decomp', args.output + '/new2', ['fuzzy'])

    if 'form' in args.steps:
        analyze_word_formation(words, args.output)

    if 'discover' in args.steps:
        compsfile = args.output + '/new-decomp'
        outputname = args.output + '/new2'
        recipes = read_recipes(compsfile)
        e2f = make_e2f_dict(f2e)

        os.makedirs(outputname, exist_ok=True)
        discover_new_compounds(recipes, f2e, e2f, outputname, ['fuzzy'], singlelang=args.singlelang)

    if 'makeqsubs' in args.steps:
        for i, lang in enumerate(sorted(f2e.keys())):
            write_qsub_array(i + 1, lang)
        write_qsub_script(len(f2e))



def write_qsub_script(num_array_scripts):
    with open('discocomp.sh', 'w') as fout:
        print('#$ -t 1-' + str(num_array_scripts), file=fout)
        print('#$ -cwd', file=fout)
        # print('#$ -M wwu37@jhu.edu', file=fout)
        # print('#$ -m e', file=fout)
        print('#$ -l mem_free=5G,ram_free=5G', file=fout)
        print('#$ -o outerr/output', file=fout)
        print('#$ -e outerr/error', file=fout)
        print('sh qsub/$SGE_TASK_ID > outerr/${JOB_ID}.${SGE_TASK_ID}.o 2> outerr/${JOB_ID}.${SGE_TASK_ID}.e', file=fout)


def write_qsub_array(index, lang):
    with open('qsub/' + str(index), 'w') as fout:
        print('source /home/wwu/softw/anaconda3/envs/py36/bin/activate py36', file=fout)
        print('./analyze.py -i out/output/new-all.tsv -o out/output -d dictionaries/wiktionary.txt --steps discover --single ' + lang, file=fout)


if __name__ == '__main__':
    main()

    # args = parseargs()
    # F2E = find_compounds.load_dict(args.dict)

    # if args.command == 'postag':
    #     with open(args.output, 'w') as f:
    #         for concept in sorted(di.concept2comps.keys()):
    #             tag = postag(concept, redo=True)
    #             print(concept + '\t' + tag, file=f)
