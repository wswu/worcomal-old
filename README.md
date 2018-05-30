# worcomal

Modeling word compounding across languages.

# Demo

To run

    ./experiment.py demo

Sample input:

To single foreign language

    f nb hospital (to single foreign language)

To all foreign languages

    f * hospital

To English

    e nb sykehus
    e de maladehaus

## Compound Discovery

To recreate the dataset:

First, find compounds with simple concatenation.

    ./find_compounds.py exact compounds --dict wiktionary

Create a single tsv file `all.tsv`.

    ./analyze.py -i compounds -o out/output -d dictionaries/wiktionary.txt --steps group

Generate a testset

    ./experiment.py maketestset > testset

Decompose compounds. This creates `decomp`.

    ./analyze.py -i out/output/all.tsv -o out/output -d dictionaries/wiktionary.txt --steps decomp --testset testset

Augment compounds using glue and drop left mechanisms. This creates several `new-*` files.

    ./analyze.py -i out/output/all.tsv -o out/output -d dictionaries/wiktionary.txt --steps augment --num_thread 10

Augment compounds using 2 character glue. This creates several `new2-*` files.

    ./analyze.py -i out/output/new-all.tsv -o out/output -d dictionaries/wiktionary.txt --steps augment2 --num_thread 10

Compute counts for compounding mechanism.

    ./analyze.py -i out/output/new-all.tsv -o out/output/new-form.txt -d dictionaries/wiktionary.txt --steps form

# Experiments

To recreate experiments:

e2f

    ./experiment.py --step e2f > exp.e2f

f2e (if more than 10, the job gets killed for some reason)

    ./experiment.py --step f2e > f2e.1

