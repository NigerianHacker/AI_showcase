import copy
import re
import random
import numpy as np
from nltk.corpus import stopwords
import math
import simplemma
import pandas as pd
from collections import Counter


# P(word |class )=(word_count_in_class + 1) / (total_words_in_class+total_unique_words_in_all_classes)


def load_csv(filepath):          #spam is 1 ham is 0
    dic={"spam":[],"ham":[]}
    with open(filepath,'r',encoding='UTF-8') as f:
        for val in f.readlines():
            val = val.lower()
            val = re.sub(r'http\S+', ' ', val)
            val = re.sub(r'[^\w\s]', ' ', val)
            val = val.replace("\n","")
            val = val.replace('^[0-9]*$',' ')
            temp = re.split("\s",val,1)
            dic[temp[0]].append(temp[1])
    dic[1] = dic.pop('spam')
    dic[0] = dic.pop('ham')
    all_messages = [(words, k) for k,v in dic.items() for words in v]
    random.shuffle(all_messages)
    all_messages=all_messages[0:1500]        #This is to reduce dataset for faster testing
    return [i[0] for i in all_messages],np.array([i[1] for i in all_messages])            #converted to np for speed purposes (I think?)







X,Y=load_csv("SMSSpamCollection")

class BM():
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y

    def test_train_split(self,train_size=0.8):  #also known as fit
        x = round(len(self.X) * train_size)
        X_test, X_train =self.X[x:], self.X[:x]
        y_test,y_train = self.Y[x:], self.Y[:x]
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test


    def falsifier(self):
        new= np.array([int(not i) for i in self.y_test])
        self.y_test=new


    def tokenize(self,doc):
        token_sequence=[]
        for i in doc:
            x = "".join(i).split()
            y = [simplemma.lemmatize(i,lang='en') for i in x if i not in stopwords.words("english")]
            token_sequence.append(y)        #BUT YOURE NOT HERE and other words are just FULL of stopwords for ham
            '''else:
                token_sequence.append("<empty>")'''
        return token_sequence   #returns something like [[you,are,gay],[why,are,you,gay]]




    def vectorizer(self,docs):
        length = len(docs)
        tf_matrix = np.zeros((length, len(self.vocab)))
        for i, sentence in enumerate(docs):
            if len(sentence) == 0:      #put in here instead in tokenizer
                continue
            for j, term in enumerate(self.vocab):
                tf_matrix[i, j] = sentence.count(term)/len(sentence)
        # Calculate the inverse document frequency
        idf_matrix = np.zeros((1, len(self.vocab)))
        for j, term in enumerate(self.vocab):
            doc_count = sum([1 for sentence in docs if term in sentence])
            idf_matrix[0, j] = math.log(length /(1+ doc_count))
        # Calculate the tf-idf matrix
        tfidf_matrix = tf_matrix * idf_matrix
        #print(docs[0])
        #print(np.unique(tfidf_matrix[0]))
        return tfidf_matrix     #FIRST implementation. WORKS. but not the way i want it to. and also really clunky idfk whats what


    def train(self):
        self.train_token_sequence = self.tokenize(self.X_train)
        self.train_flat_token_sequence = [item for sublist in self.train_token_sequence for item in sublist]
        self.vocab=set(self.train_flat_token_sequence)
        tfidf= self.vectorizer(self.train_token_sequence)       #tfidf for x_train
        a,b = np.where(self.y_train == 0), np.where(self.y_train == 1) #a=ham, b = scam
        self.tfidf = {0:tfidf[a],1:tfidf[b]}

        #print(self.scam_matrix.shape)
        #print(self.ham_matrix.shape)
        #print(len(self.y_train))


    def bow(self):
        a= np.where(self.y_train == 0)[0]
        b = np.where(self.y_train == 1)[0]
        ham_bow = [self.train_token_sequence[i] for i in a]
        spam_bow= [self.train_token_sequence[i] for i in b]
        flat_ham= [item for sublist in ham_bow for item in sublist]
        flat_spam= [item for sublist in spam_bow for item in sublist]
        dict_ham =  Counter(flat_ham)#dict.fromkeys(self.vocab,0)
        dict_spam =  Counter(flat_spam)#copy.copy(dict1)
        dict_total = {0:dict_ham, 1:dict_spam}
        token_sequence = self.tokenize(self.X_test)
        scam_chances = np.count_nonzero(self.y_train) / self.y_train.size
        p_class = {1: scam_chances, 0: 1 - scam_chances}
        y_pred=[]      #each term in this list is 1 or 0 depending for each sentence
        for sentence in token_sequence:
            p={}
            for c in np.unique(self.y_train):
                for term in sentence:
                    p_word_given_class = {0: {}, 1: {}}
                    try:
                        p_word_given_class[c][term] = float((dict_total[c][term] + 1)/(sum(dict_total[c].values()) + len(self.vocab)))
                    except:
                        p_word_given_class[c][term] = float(1/(sum(dict_total[c].values()) + len(self.vocab)))
                log_likelihood = np.log(p_class[c]) + math.log(np.prod(np.fromiter(p_word_given_class[c].values(), dtype=float)))
                p[c] = log_likelihood
            y_pred.append(max(p, key=p.get))
        return y_pred





        #P(word |class )=(word_count_in_class + 1) / (total_words_in_class+total_unique_words_in_all_classes)



    def predict(self):
        scam_chances = np.count_nonzero(self.y_train) / self.y_train.size
        p_class = {1: scam_chances, 0: 1 - scam_chances}
        token_sequence = self.tokenize(self.X_test)
        y_pred = []


        for sentence in token_sequence:
            p = {}
            x= list(self.vocab)
            length = len(self.vocab)
            for c in np.unique(self.y_train):
                for term in sentence:
                    p_word_given_class = {0: {}, 1: {}}
                    try:
                        j = x.index(term)
                        p_word_given_class[c][term] = np.sum(self.tfidf[c][:,j], axis=0) + 1 / np.sum(self.tfidf[c]) +length
                    except:
                        p_word_given_class[c][term] = 1 / np.sum(self.tfidf[c]) + length
                log_likelihood = np.log(np.prod(np.fromiter(p_word_given_class[c].values(), dtype=float)) + np.log(p_class[c]))
                #print(log_likelihood)
                p[c] = log_likelihood
            y_pred.append(max(p, key=p.get))
        return y_pred



        #P(word |class )=(word_count_in_class + 1) / (total_words_in_class+total_unique_words_in_all_classes)


        '''p_class = {}
        for c in np.unique(self.y_train):    #probability of prior class. only 2 SPAM or HAM
            p_class[c] = np.sum(self.y_train == c) / len(self.y_train)

        tf_idf = {}
        for token in self.vocab:
            tf = counter[token] / sum(self.tf.values())
            df = doc_freq(token)
            idf = np.log(N / (df + 1))
            tf_idf[doc, token] = tf * idf
        pass'''

M=BM(X,Y)
M.test_train_split()
#M.falsifier()
M.train()
y_pred =M.predict()
accuracy = np.mean(y_pred == M.y_test)
print('TFIDF Accuracy:', accuracy)
y_pred = M.bow()
BOWaccuracy = np.mean(y_pred == M.y_test)
print(' BOW Accuracy:', BOWaccuracy)


'''
# Step 1: Load the dataset
df = pd.read_csv('spam.csv', encoding = 'ISO-8859-1')

# Step 2: Data Preprocessing
# Clean the text data, tokenize, remove stop words and stemming
# Then, create a document-term matrix using tf-idf vectorizer
tf_vectorizer = TfidfVectorizer(stop_words='english')
X = tf_vectorizer.fit_transform(df['v2'].values.astype('U'))
#X = df['v2'].values
y = df['v1'].values

# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tf= tf_vectorizer.fit_transform(X_train)
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_tf, y_train)
X_test_tf = tf_vectorizer.transform(X_test)
y_pred = naive_bayes_classifier.predict(X_test_tf)
score1 = metrics.accuracy_score(y_test, y_pred)'''


'''
# Step 4: Create a vocabulary
vocabulary = vectorizer.vocabulary_

# Step 5: Calculate the probabilities
# Calculate prior probabilities
p_class = {}
for c in np.unique(y_train):
    p_class[c] = np.sum(y_train == c) / len(y_train)

# Calculate conditional probabilities
p_word_given_class = {}
#print(X_train)
#print(pd.DataFrame(data=X_train.toarray(), columns=vocabulary).iloc[:,::2])
for c in np.unique(y_train):
    X_train_c = X_train[y_train == c]
    p_word_given_class[c] = (X_train_c.sum(axis=0) + 1) / (np.sum(X_train_c.sum(axis=0)) + len(vocabulary))

# Step 6: Implement the Naive Bayes algorithm
def predict(X):
    y_pred = []
    for x in X:
        p = {}
        for c in np.unique(y_train):
            log_likelihood = np.sum(np.log(p_word_given_class[c])) + np.log(p_class[c])
            p[c] = log_likelihood
        y_pred.append(max(p, key=p.get))
    return y_pred

# Step 7: Evaluate the model
y_pred = predict(X_test)
accuracy = np.mean(y_pred == y_test)
print('Accuracy:', accuracy)'''


















