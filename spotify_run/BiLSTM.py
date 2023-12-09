import datetime as dt
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

from WordVectors import GoogleW2V, GloveW2V
import config
from ReviewData import ReviewDataset

DEVICE = config.DEVICE
N_WORKERS = os.cpu_count() // 2


class BiLSTM(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, n_timestamps):
        super(BiLSTM, self).__init__()
        self.hidden_dims = hidden_dims
        self.n_timestamps = n_timestamps
        self.mid_timestamp = n_timestamps // 2
        self.lstm = nn.LSTM(input_dims, hidden_dims, batch_first=True, bidirectional=True)
        # self.fc1 = nn.Linear(self.n_timestamps * hidden_dims*2, hidden_dims*2)
        # self.fc1 = nn.Linear(self.n_timestamps * hidden_dims * 2, output_dims)
        self.fc1 = nn.Linear(hidden_dims * 2, 2 * output_dims)
        self.fc2 = nn.Linear(output_dims * 2, output_dims)
        self.d_out1 = nn.Dropout(0.2)
        self.d_out2 = nn.Dropout(0.2)

    def forward(self, inputs):
        lstm_out, _ = self.lstm(inputs)
        lstm_out = lstm_out[:, self.mid_timestamp: self.mid_timestamp + 1, :]
        lstm_out = lstm_out.flatten(start_dim=1)
        lstm_out = self.d_out1(lstm_out)
        # fc1_out = nn.functional.leaky_relu(self.fc1(lstm_out))
        fc1_out = nn.functional.leaky_relu(self.fc1(lstm_out))
        fc1_out = self.d_out2(fc1_out)
        out = self.fc2(fc1_out)
        return out

    def save_the_model(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.pth")
        torch.save(self.state_dict(), model_path)


class BiLSTM_Classifier:
    def __init__(self, n_class, vectorizer, hidden_dims, max_seq_len=300):
        self.n_class = n_class
        supported_types = ['tfidf', "googlew2v", "glovew2v"]
        vectorizer = vectorizer.lower()
        assert vectorizer in supported_types, f"supported types are {supported_types}"
        if vectorizer == "tfidf":
            self.vectorizer = TfidfVectorizer()
            self.bilstm = BiLSTM(1, hidden_dims, self.n_class, max_seq_len).to(DEVICE)
        elif vectorizer == "googlew2v":
            self.vectorizer = GoogleW2V()
            self.bilstm = BiLSTM(300, hidden_dims, self.n_class, max_seq_len).to(DEVICE)
        elif vectorizer == "glovew2v":
            self.vectorizer = GloveW2V()
            self.bilstm = BiLSTM(300, hidden_dims, self.n_class, max_seq_len).to(DEVICE)
        else:
            raise RuntimeError("Allowed types are tfidf, googlew2v, GloveW2V")

        self.hidden_dims = int(hidden_dims)
        self.max_seq_len = int(max_seq_len)
        self.criterion = nn.CrossEntropyLoss()

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

    def __check_accuracy(self, y_pred, y_true):
        pass

    def __match_preds(self, y_pred, y_true):
        max_idxs = torch.argmax(y_pred, dim=1)
        matches = list((max_idxs == y_true).cpu().numpy())
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
            lr=lr,
            weight_decay=1e-9
        )

        # setting LR scheduler
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.1,
            patience=3,
            threshold=1e-2,
            verbose=True
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
                batch_label = batch_label.type(torch.LongTensor).squeeze(dim=1).to(DEVICE)
                batch_op = self.bilstm(batch_data)
                loss = self.criterion(batch_op, batch_label)    # 64,1 to 64
                loss.backward()
                optimizer.step()
                train_batch_losses.append(loss.item())
                train_preds_match += self.__match_preds(batch_op, batch_label)

            with torch.no_grad():
                for batch_data, batch_label in val_dataloader:
                    batch_data = batch_data.to(DEVICE)
                    batch_label = batch_label.type(torch.LongTensor).squeeze(dim=1).to(DEVICE)
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
            lr_scheduler.step(val_e_acc)

        return train_epoch_loss, train_epoch_acc, val_epoch_loss, val_epoch_acc


if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split

    DATASET_PATH = config.DATASET_PATH
    SEQ_LEN = 80
    VECTORIZETR = "glovew2v"

    reviews_df = pd.read_csv(DATASET_PATH)
    # making the column names lowercase
    reviews_df.columns = [colname.lower() for colname in reviews_df.columns]
    reviews_train, reviews_test = train_test_split(reviews_df,
                                                   train_size=0.8,
                                                   random_state=711
                                                   )

    n_classes = len(np.unique(reviews_train["rating"]))
    b_clf = BiLSTM_Classifier(n_classes, VECTORIZETR, 64, SEQ_LEN)

    trainDS = ReviewDataset(
        reviews_train["review"],
        reviews_train['rating'],
        VECTORIZETR,
        SEQ_LEN,
        isReg=False
    )
    valDS = ReviewDataset(
        reviews_test["review"],
        reviews_test['rating'],
        VECTORIZETR,
        SEQ_LEN,
        isReg=False
    )

    b_op = b_clf.trainBiLSTM(
        trainDS,
        valDS,
        epochs=100,
        lr=1e-4,
        optimizer="adam",
        batch_size=64
    )
    print(b_op)
    b_clf.bilstm.save_the_model(config.MODEL_DIR)

# 0.001
# 0.01