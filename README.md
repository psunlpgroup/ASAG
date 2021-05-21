# ASAG
Automatic Short Answer Grading


## Python Environment

Python version: 3.6.10

If you use Anaconda you can follow these steps:

    conda create -n asag_pyenv python=3.6.10 

    conda activate asag_pyenv 

If you already have a python 3.6 environment, you can ignore this step. 
Then install the requirements:

    pip install -r requirements.txt 

    python -m spacy download en

## Download pre-trained word vectors


Down load the Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download): [glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip)

Unzip it and put it under ./model folder

## experiment running 

You can run the test example data by: 

    python train.py