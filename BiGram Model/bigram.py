import re 
import random
import sys 

class LanguageModel():
    def __init__(self):
        self.bigrams = {}

    def pre_process(self, corpus):
        lines = list(corpus.splitlines())
        
        for index in range(len(lines)):
            if len(lines[index]) == 0:
                continue
            lines[index] = re.sub(r'[^\w\s]', '', lines[index])
            # lines[index] = '<s> ' + lines[index]
            lines[index] = re.sub(r'[0-9]', '', lines[index])
            lines[index] = lines[index].lower()
            # lines[index] += ' </s> '
    
        token_sequence = ''.join(lines).split()
        return token_sequence
    
    
    def train(self, corpus):
        token_sequence = self.pre_process(corpus)
        vocab = set(token_sequence)
        bigrams = {}
        
        for i in range(len(token_sequence)-1):
            curent_token = token_sequence[i]
            next_token = token_sequence[i + 1]
      
            if curent_token not in bigrams:
                bigrams[curent_token] = {'internal_count' : 0} # init the current word to 0
                
            bigrams[curent_token]['internal_count'] += 1 # now increment because we have found a token (an instance) of a word 
            
            if next_token not in bigrams[curent_token]: # same thing for the next word (token of a word)
                bigrams[curent_token][next_token] = 0 
                
            bigrams[curent_token][next_token] += 1
        
        self.bigrams = bigrams
        # print(bigrams)
    
    def bigram_probaility(self, bigram):
        # print("Current Bigram:", bigram)
        # print("Current Bigram[0]:", bigram[0])
        # print("Current Bigram[1]:", bigram[1])
        # print("Self Bigrams:", self.bigrams)
        # bigrams = self.count_bigrams(corpus)
        try:
            # P(w2|w1) = C(w1,w2) / C(w1)
            probability = self.bigrams[bigram[0]][bigram[1]] / (self.bigrams[bigram[0]]['internal_count']) 
            return probability # bigrams[bigram[0]][bigram[1]]/(bigrams[bigram[0]]['internal_count'])
        except KeyError as e:
             return 0 # (f'The word {e} is not part of our corpus, or this bigram doesn\'t co-occur')
    
    
    def perplexity(self, corpus): # PP(W) = P(w1,w2,...,wn) ** -1/N 
        token_sequence = self.pre_process(corpus) # W
        num_bigrams = len(token_sequence) # N
        perplexity = 1 # product starts from 1 
        for i in range(num_bigrams - 1):
            bigram = [token_sequence[i],token_sequence[i + 1]] # wn
            probability = self.bigram_probaility(bigram) # P(w)

            # here we should add smooting because somethimes we end up with zero probability 
            if probability == 0:
                probability = sys.float_info.min
                
            perplexity *= probability ** (- 1 / num_bigrams) # PP(W) 
            
            # print(f'{bigram} --> {probaility}')
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


model = LanguageModel()

corpus_jungle = (open("jungle_book.txt")).read()
corpus_wiki_test = (open('wiki.test.raw')).read()
corpus_wiki_train = (open('wiki.train.raw')).read()

model.train(models_corpus)
model.perplexity(models_corpus)

model.generate_sentence()
