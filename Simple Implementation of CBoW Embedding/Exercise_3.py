import re
import string
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as px


def load_csv(filepath):
    with open(filepath,'r' ,encoding='utf-8') as x:
        a = x.read().splitlines()
        return a

def preprocess(data):
    for i in range(len(data)):
        data[i] = data[i].lower()      #lowercase
        data[i] = re.sub(r'[ \t]+', ' ', data[i])       #remove double spaces etc.
        data[i] = data[i].translate(str.maketrans('', '', string.punctuation))      #removes punctuation
    return data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, corpus, window_size):
        self.corpus = corpus
        self.window = window_size
        self.tokenized_sentences = [sentence.split() for sentence in corpus]
        self.vocab = set([item for sublist in self.tokenized_sentences for item in sublist])
        self.vocab_size = len(self.vocab)
        self.window_size = window_size
        self.word_idx = {word: idx for idx, word in enumerate(self.vocab)}  # put the word and converts it into an int
        self.idx_word = list(self.word_idx.keys())  # put the int and converts it into word (not rly needed,
        # just for testing purposes)
        self._initialize()

    def _initialize(self):
        self.X = []
        self.Y = []
        for sentence in self.tokenized_sentences:
            sentence_indices = [self.word_idx[word] for word in sentence]  # converts [hi,i,am,a,boy] into [3,5,6,7,3]
            sentence_length = len(sentence_indices)
            target = [(j, sentence_indices[j]) for j in
                      range(self.window_size, sentence_length - self.window_size)]  # tuple of target_idx and target
            for i in target:
                context = [sentence_indices[i[0] - 2], sentence_indices[i[0] - 1], sentence_indices[i[0] + 1],
                           sentence_indices[i[0] + 2]]
                self.X.append(torch.tensor(context,dtype=torch.long))
                self.Y.append(torch.tensor(i[1],dtype= torch.long))

    def __getitem__(self, idx):
        con = self.X
        tar = self.Y

        return con[idx], tar[idx]

    def __len__(self):
        return len(self.Y)

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embedding_matrix = nn.Parameter(torch.randn(vocab_size, embedding_dim))
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        #print(self.embedding_matrix)
        embedded = self.embedding_matrix[x]
        #print(embedded)
        summed = torch.sum(embedded, dim=1)
        #print(summed)
        out = self.linear(summed)
        log_probs = nn.functional.log_softmax(out, dim=1)
        return log_probs


data = load_csv('Tiny.raw')
corpus = preprocess(data)
ds = Dataset(corpus,2)
batch = 1
data_loader = DataLoader(ds, batch_size=batch)
embedding_dim = 10
lr = 0.001
epochs = 80
model = CBOW(len(ds.vocab),embedding_dim)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
for epoch in range(epochs):
    for x,y in data_loader:
        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_function(outputs, y)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss:{loss.item()}')

embedding_weights = model.embedding_matrix.data.numpy()
tsne = TSNE(random_state = 0, n_iter = 1000, n_components=3)
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters)
cluster_labels = kmeans.fit_predict(embedding_weights)
#df['cluster_labels'] = cluster_labels
embeddings3d = tsne.fit_transform(embedding_weights)
df = pd.DataFrame(embeddings3d, columns = ['0','1','2'])
df['word'] = ds.idx_word
print(df)

'''
embeddingsword = pd.DataFrame()
embeddingsword['word'] = ds.idx_word
embeddingsword.to_csv('large_word.tsv', sep="\t", index=False)
embeddingsdf = pd.DataFrame()
embeddingsdf['1'] = embedding_weights[:,0]
embeddingsdf['2'] = embedding_weights[:,1]
embeddingsdf['3'] = embedding_weights[:,2]
embeddingsdf['4'] = embedding_weights[:,3]
embeddingsdf['5'] = embedding_weights[:,4]
embeddingsdf['6'] = embedding_weights[:,5]
embeddingsdf['7'] = embedding_weights[:,6]
embeddingsdf['8'] = embedding_weights[:,7]
embeddingsdf['9'] = embedding_weights[:,8]
embeddingsdf['10'] = embedding_weights[:,9]
embeddingsdf = embeddingsdf.reset_index(drop=True)
embeddingsdf.to_csv('large_emb.tsv', sep="\t", index=False)
'''




# Plot the embeddings with cluster labels
fig = px.scatter_3d(df, x='0', y='1', z='2', color=cluster_labels,text = 'word')
#plt.scatter(embeddings3d[:, 0], embeddings2d[:, 1], c=cluster_labels, alpha = 0.1)
fig.show()
