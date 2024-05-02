import copy
import string
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

with open("ptbdataset/ptbdataset/ptb.train.txt", "r") as x:
    a= x.read()
with open("ptbdataset/ptbdataset/ptb.test.txt", "r") as x:
    b= x.read()
'''with open("wikitext-2-raw-v1/wikitext-2-raw/wiki.test.raw", "r",encoding= "utf8") as x:
    c = x.read()
with open("ptbdataset/ptbdataset/ptb.valid.txt", "r") as x:
    d=x.read()
'''
class LM():
    def __init__(self,n):
        pass

    def pre_process(self,data):
        lines = list(data.splitlines())
        for i in range(len(lines)):
            if len(lines[i]) == 0:
                continue
            lines[i] = ' <s>' + lines[i]
            lines[i] = lines[i].lower()
            lines[i] += ' </s> '
        token_sequence = "".join(lines).split()
        return token_sequence


    def train(self,data):
        token_sequence = self.pre_process(data)
        self.vocab = set(token_sequence)
        bigrams={}
        for i in range(len(token_sequence)-1):
            if token_sequence[i] not in bigrams:
                bigrams[token_sequence[i]] ={'internal_count':0}
            if token_sequence[i+1] not in bigrams[token_sequence[i]]:
                bigrams[token_sequence[i]][token_sequence[i + 1]] = 0
            bigrams[token_sequence[i]]['internal_count'] += 1
            bigrams[token_sequence[i]][token_sequence[i+1]] += 1
        self.bigrams = bigrams
        #print(self.bigrams['no']['it'])





    def prob(self,bigram):#bigram is a tuple of (the,asbestos)    p(no <s>) = c(<s> no)/c(<s>)
        probs = self.bigrams[bigram[0]][bigram[1]]/(self.bigrams[bigram[0]]['internal_count'])
        return probs



    def perplexity(self,data):   #P(w1w2 ...wN)**(-1/n)
        token_sequence = self.pre_process(data)
        perp = 1
        for i in range(len(token_sequence)-1):
            bigram = [token_sequence[i],token_sequence[i+1]]
            bi_prob = self.prob(bigram)
            #print(bi_prob,bigram)
            perp = perp * (bi_prob ** (-1/(len(token_sequence))))
        return perp

model=LM(2)
model.train(a)
print(model.bigrams)
print(model.perplexity(a))
