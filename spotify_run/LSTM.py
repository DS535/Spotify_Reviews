#library imports
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
import spacy
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import string
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import config

class ReviewsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx][0].astype(np.int32)), self.y[idx], self.X[idx][1]
    
class LSTM_1_layer(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim) :
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 5)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, l):
        x = self.embeddings(x)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])
    
#2-layer LSTM
class LSTM_2_layer(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 5)
        self.dropout = nn.Dropout(0.2)
        self.num_layers = num_layers
        
    def forward(self, x, l):
        x = self.embeddings(x)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        for i in range(self.num_layers-1):
            lstm_out, (ht, ct) = self.lstm(lstm_out)
        return self.linear(ht[-1])

    
def train_model(model, epochs=10, lr=0.001):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    print("Model Training Started..")
    for i in range(epochs):
        model.train()
        
        sum_loss = 0.0
        total = 0
        correct = 0
        for x, y, l in train_dl:
            x = x.long()
            y = y.long()
            y_pred = model(x, l)
            pred = torch.max(y_pred, 1)[1]
            optimizer.zero_grad()
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()
            correct += (pred == y).float().sum()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
        val_loss, val_acc, conf_matrix = validation_metrics(model, val_dl)
        train_losses.append(sum_loss/total)
        train_accs.append(correct/total)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        if i % 5 == 1:
            print("train loss %.3f, val loss %.3f, train acuracy %.3f, val accuracy %.3f" % (sum_loss/total, val_loss, correct/total, val_acc))
            print("Confusion Matrix:")
            print(conf_matrix)   
    # Plot the training and validation loss and accuracy curves
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    ax[0].plot(range(1, epochs+1), train_losses, label='Training')
    ax[0].plot(range(1, epochs+1), val_losses, label='Validation')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[1].plot(range(1, epochs+1), train_accs, label='Training')
    ax[1].plot(range(1, epochs+1), val_accs, label='Validation')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

from sklearn.metrics import mean_squared_error, confusion_matrix

def validation_metrics(model, valid_dl):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_rmse = 0.0
    all_predictions = []
    all_targets = []
    for x, y, l in valid_dl:
        x = x.long()
        y = y.long()
        y_hat = model(x, l)
        loss = F.cross_entropy(y_hat, y)
        pred = torch.max(y_hat, 1)[1]
        all_predictions.extend(pred.tolist())
        all_targets.extend(y.tolist())
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]
        #sum_rmse += np.sqrt(mean_squared_error(pred.cpu().numpy(), y.unsqueeze(-1).cpu().numpy()))*y.shape[0]
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    return sum_loss/total, correct/total, conf_matrix

#Preprocessing functions
tok = spacy.load('en_core_web_sm')
def tokenize (text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]') # remove punctuation and numbers
    nopunct = regex.sub(" ", text.lower())
    return [token.text for token in tok.tokenizer(nopunct)]

def encode_sentence(text, vocab2index, N=70):
    tokenized = tokenize(text)
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    return encoded, length

if __name__ == "__main__":
    DATASET_PATH = config.DATASET_PATH
    reviews = pd.read_csv(DATASET_PATH)
    reviews.columns=['Time_submitted', 'review', 'Rating', 'Total_thumbsup', 'Reply']
    #keeping only relevant columns and calculating sentence lengths
    reviews = reviews[['review', 'Rating']]
    reviews.columns = ['review', 'rating']
    reviews['review_length'] = reviews['review'].apply(lambda x: len(x.split()))
    #updating labels
    zero_numbering = {1:0, 2:1, 3:2, 4:3, 5:4}
    reviews['rating'] = reviews['rating'].apply(lambda x: zero_numbering[x])
    #count number of occurences of each word
    counts = Counter()
    for index, row in reviews.iterrows():
        counts.update(tokenize(row['review']))
    #deleting infrequent words
    print("num_words before:",len(counts.keys()))
    for word in list(counts):
        if counts[word] < 2:
            del counts[word]
    print("num_words after:",len(counts.keys()))
    #creating vocabulary
    vocab2index = {"":0, "UNK":1}
    words = ["", "UNK"]
    for word in counts:
        vocab2index[word] = len(words)
        words.append(word)

    reviews['encoded'] = reviews['review'].apply(lambda x: np.array(encode_sentence(x,vocab2index )))
    X = list(reviews['encoded'])
    y = list(reviews['rating'])
    from sklearn.model_selection import train_test_split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    train_ds = ReviewsDataset(X_train, y_train)
    valid_ds = ReviewsDataset(X_valid, y_valid)
    batch_size = 5000
    vocab_size = len(words)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(valid_ds, batch_size=batch_size)
    #Training 1-Layer LSTM with 64 hidden-vector dimension.
    model_64 =  LSTM_1_layer(vocab_size, 50, 64)
    train_model(model_64, epochs=30, lr=0.01)
    print("Training for LSTM-1-Layer completed!")
    #Training 2-Layer LSTM with 256 hidden-vector dimension.
    model_64 =  LSTM_2_layer(vocab_size, 64, 64)
    train_model(model_64, epochs=30, lr=0.01)
    print("Training for LSTM-2-Layer completed!")