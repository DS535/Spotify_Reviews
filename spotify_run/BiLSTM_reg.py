import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.feature_extraction.text import TfidfVectorizer
from WordVectors import GoogleW2V
from sklearn.preprocessing import OneHotEncoder

import config
from ReviewData import ReviewDataset

DEVICE = config.DEVICE
N_WORKERS = os.cpu_count() // 2


class BiLSTM(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, n_timestamps):
        super(BiLSTM, self).__init__()
        self.hidden_dims = hidden_dims
        self.n_timestamps = n_timestamps
        self.lstm = nn.LSTM(input_dims, hidden_dims, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(self.n_timestamps * hidden_dims*2, hidden_dims*2)
        self.fc2 = nn.Linear(hidden_dims * 2, output_dims)

    def forward(self, inputs):
        lstm_out, _ = self.lstm(inputs)
        lstm_out = lstm_out.flatten(start_dim=1)
        fc1_out = nn.functional.leaky_relu(self.fc1(lstm_out))
        out = torch.sigmoid(self.fc2(fc1_out))
        # print(out)
        return out


class BiLSTM_Reg:
    def __init__(self, n_class, vectorizer, hidden_dims, max_seq_len=300):
        self.n_class = n_class
        supported_types = ['tfidf', "googlew2v"]
        vectorizer = vectorizer.lower()
        assert vectorizer in supported_types, f"supported types are {supported_types}"
        if vectorizer == "tfidf":
            self.vectorizer = TfidfVectorizer()
            self.bilstm = BiLSTM(1, hidden_dims, 1, max_seq_len).to(DEVICE)
        elif vectorizer == "googlew2v":
            self.vectorizer = GoogleW2V()
            self.bilstm = BiLSTM(300, hidden_dims, 1, max_seq_len).to(DEVICE)
        else:
            raise RuntimeError("Allowed types are tfidf and googlew2v")

        self.hidden_dims = int(hidden_dims)
        self.max_seq_len = int(max_seq_len)
        self.criterion = nn.MSELoss()
        self.__divisor = 1 / self.n_class

    def __tf_idf_vectorize(self, txt_data):
        txt_data_trnsformed = self.vectorizer.fit_transform(txt_data)
        tf_docs = []
        for doc in txt_data_trnsformed:
            tf_doc = doc.data
            # trim if len > self.max_seq_len else I have to append with zeros
            if len(tf_doc) > self.max_seq_len:
                tf_doc = tf_doc[:self.max_seq_len]
            else:
                zero_len = self.max_seq_len - len(tf_doc)
                tf_doc = list(tf_doc) + [0] * zero_len
            tf_docs.append(tf_doc)

        tf_docs = np.array(tf_docs).reshape((len(tf_docs), self.max_seq_len, 1))
        return tf_docs

    def __vectorize(self, text_data):
        if isinstance(self.vectorizer, TfidfVectorizer):
            return self.__tf_idf_vectorize(text_data)
        elif isinstance(self.vectorizer, GoogleW2V):
            return self.vectorizer.convert_to_vec(text_data)

    def __check_accuracy(self, y_pred, y_true):
        pass

    def __match_preds(self, y_pred, y_true):
        y_pred_cls = (y_pred / self.__divisor - 1).to(torch.long)
        y_true_cls = ((y_true + 1e-6) * self.n_class).to(torch.long)
        matches = list((y_pred_cls == y_true_cls).cpu().numpy())
        return matches

    def trainBiLSTM(
            self,
            train_dataset,
            val_dataset,
            epochs,
            lr,
            optimizer,
            batch_size,
            lr_scheduler=None
    ):
        assert epochs >= 1
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=2 * batch_size
        )

        # setting the optimizer
        optimizer = Adam(
            self.bilstm.parameters(),
            lr=lr
        )

        train_epoch_loss = []
        val_epoch_loss = []
        train_epoch_acc = []
        val_epoch_acc = []
        for epoch in range(1, epochs + 1):
            train_batch_losses = []
            train_preds_match = []
            val_batch_losses = []
            val_preds_match = []

            print(f"epoch: {epoch}")
            for batch_data, batch_label in train_dataloader:
                optimizer.zero_grad()
                # print(f"batch data shape: {batch_data.shape}")
                batch_data = batch_data.to(DEVICE)
                batch_label = batch_label.to(DEVICE)
                batch_op = self.bilstm(batch_data)
                loss = self.criterion(batch_op, batch_label)
                loss.backward()
                optimizer.step()
                train_batch_losses.append(loss.item())
                train_preds_match += self.__match_preds(batch_op, batch_label)

            with torch.no_grad():
                for batch_data, batch_label in val_dataloader:
                    batch_data = batch_data.to(DEVICE)
                    batch_label = batch_label.to(DEVICE)
                    batch_op = self.bilstm(batch_data)
                    loss = self.criterion(batch_op, batch_label)
                    val_batch_losses.append(loss.item())
                    val_preds_match += self.__match_preds(batch_op, batch_label)

            # calculate epoch stats
            train_e_loss = np.mean(train_batch_losses)
            val_e_loss = np.mean(val_batch_losses)
            train_e_acc = sum(train_preds_match) / len(train_preds_match)
            val_e_acc = sum(val_preds_match) / len(val_preds_match)
            print(f"Train Loss: {train_e_loss}\t Train_acc: {train_e_acc}")
            print(f"Val Loss: {val_e_loss}\t Val_acc: {val_e_acc}")
            print()

            train_epoch_loss.append(train_e_loss)
            val_epoch_loss.append(val_e_loss)
            train_epoch_acc.append(train_e_acc)
            val_epoch_acc.append(val_e_acc)

        return train_epoch_loss, train_epoch_acc, val_epoch_loss, val_epoch_acc


if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split

    DATASET_PATH = config.DATASET_PATH

    reviews_df = pd.read_csv(DATASET_PATH)
    # making the column names lowercase
    reviews_df.columns = [colname.lower() for colname in reviews_df.columns]
    reviews_train, reviews_test = train_test_split(reviews_df,
                                                   train_size=0.8,
                                                   random_state=711
                                                   )

    n_classes = len(np.unique(reviews_train["rating"]))
    b_clf = BiLSTM_Reg(n_classes, "googlew2v", 64, 150)

    trainDS = ReviewDataset(
        reviews_train["review"],
        reviews_train['rating'],
        "googlew2v",
        150
    )
    valDS = ReviewDataset(
        reviews_test["review"],
        reviews_test['rating'],
        "googlew2v",
        150
    )

    b_op = b_clf.trainBiLSTM(
        trainDS,
        valDS,
        epochs=30,
        lr=0.0005,
        optimizer="adam",
        batch_size=32
    )
    print(b_op)

# 0.001
# 0.01