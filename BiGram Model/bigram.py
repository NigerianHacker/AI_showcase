import re 
import random
import sys 

class LanguageModel():
    def __init__(self):
        self.bigrams = {}
        self.token_sequence = []
        self.vocab_size = 0
        self.vocab = set()

    def pre_process(self, corpus):
        lines = list(corpus.splitlines())
        
        for index in range(len(lines)):
            if len(lines[index]) == 0:
                continue
            lines[index] = re.sub(r'[^\w\s]', '', lines[index]) # reove non-words like punctiuation and digits 
            lines[index] = ' <s>' + lines[index]
            lines[index] = re.sub(r'[0-9]', '', lines[index])
            lines[index] = lines[index].lower()
            lines[index] += ' </s> '
    
        token_sequence = ''.join(lines).split()
        self.token_sequence = token_sequence
        return token_sequence
    
    
    def train(self, corpus):
        token_sequence = self.pre_process(corpus)
        vocab = set(token_sequence)
        self.vocab_size = len(vocab)
        self.vocab = vocab
        bigrams = {}
        
        for i in range(len(token_sequence)-1):
            curent_token = token_sequence[i]
            next_token = token_sequence[i + 1]
      
            if curent_token not in bigrams:
                # init the current word to 1 beacsue we use add-one smoothing 
                bigrams[curent_token] = {'internal_count' : 1}
            bigrams[curent_token]['internal_count'] += 1 # now increment because we have found a token (an instance) of a word 
            
            if next_token not in bigrams[curent_token]: # same thing for the next word (token of a word)
                bigrams[curent_token][next_token] = 1
            bigrams[curent_token][next_token] += 1
        
        self.bigrams = bigrams
    
    def bigram_probaility(self, bigram):
        # print("Current Bigram:", bigram)
        # print("Current Bigram[0]:", bigram[0])
                                                                              #   bigrams = {'the': {'internal_count': 2,
                                                                              #              'project': 1}
        # print("Current Bigram[1]:", bigram[1])
        # print("Self Bigrams:", self.bigrams)
                                                                              #   bigram = ['the', 'project']
        # bigrams = self.count_bigrams(corpus)

        # ### linear interpolation #################
        # try:
        #     a = 1 / 2
        #     prob_inter = a * (self.bigrams[bigram[0]]['internal_count'] / self.vocab_size) + \
        #     (1 - a) * (self.bigrams[bigram[0]][bigram[1]] / self.bigrams[bigram[0]]['internal_count'])
        #     return prob_inter
        # except KeyError:
        #     return 1 / self.vocab_size
        # ########################################## ^ (174.5604163134672 195.326091981654) ^
        
        try:
            # P(w2|w1) = C(w1,w2) / C(w1)
            probability = (self.bigrams[bigram[0]][bigram[1]]) / (self.bigrams[bigram[0]]['internal_count'])
            return probability
        except KeyError:
            try:
                unigram_probability = (self.bigrams[bigram[0]]['internal_count']) / self.vocab_size
                return unigram_probability
            except KeyError:
                return 1 / self.vocab_size
        ############################################ ^ (140.59207466850373 147.06623612019703) ^
    
    def perplexity(self, corpus): # PP(W) = P(w1,w2,...,wn) ** -1 / N 
        token_sequence = self.pre_process(corpus) # W
        num_bigrams = len(token_sequence) # N
        perplexity = 1 # product starts from 1 

        for i in range(num_bigrams - 1):
            bigram = [token_sequence[i],token_sequence[i + 1]] # wn
            probability = self.bigram_probaility(bigram) # P(w)
            perplexity *= probability ** (- 1 / num_bigrams) # PP(W) 
            #print(f'{bigram} --> {probability}')
        return perplexity
    
    ### Additional Experiments (5 additional points - Optional)
    def generate_sentence(self):
        sentence = ['<s>']
        current_token = '<s>'
    
        while current_token != '</s>':
            if current_token not in self.bigrams or not self.bigrams[current_token]:
                break
            
            next_tokens = list(self.bigrams[current_token].keys())
            next_tokens.pop(0) # remove our internal_counter 
            probabilities = [self.bigram_probaility((current_token, next_token)) for next_token in next_tokens]
            
            next_token = random.choices(next_tokens, weights=probabilities)[0]
            sentence.append(next_token)
            current_token = next_token
            # print(next_token)

        generated_sentence = ' '.join(sentence)
        return generated_sentence

corpus_jungle = (open("jungle_book.txt")).read()

corpus_wiki_train = (open('Datasets_and_resources/wiki.train.raw')).read()
corpus_wiki_valid = (open('Datasets_and_resources/wiki.valid.raw')).read()
corpus_wiki_test = (open('Datasets_and_resources/wiki.test.raw')).read()

# models_corpus = open('models.txt').read()
# под_игото = (open('Ivan_Vazov_-_Pod_igoto_-_1773-b.txt')).read()

model = LanguageModel()
model.train(corpus_wiki_train)

print(model.perplexity(corpus_wiki_valid), model.perplexity(corpus_wiki_test))

# model.generate_sentence()
