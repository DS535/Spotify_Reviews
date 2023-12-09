import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import EngPreprocessing
import WordVectors
import config


class ReviewDataset(Dataset):
    def __init__(self, reviews, labels, vectorizer, length=100, language="english", isReg=True):
        print("object creation started")
        assert len(reviews) == len(labels)
        if language == "english":
            preproc = EngPreprocessing()
        else:
            preproc = None

        self.reviews = [preproc.preprocess(review) for review in reviews]
        self.labels = labels
        self.isReg = isReg

        # dealing with vectorizer
        assert vectorizer in ["googlew2v", "tfidf", "glovew2v"]
        if vectorizer == "tfidf":
            self.vectorizer = TfidfVectorizer()
            self.vectorizer.fit(self.reviews)
        elif vectorizer == "googlew2v":
            self.vectorizer = WordVectors.GoogleW2V()
        elif vectorizer == "glovew2v":
            self.vectorizer = WordVectors.GloveW2V()
        self.n_classes = len(np.unique(labels))
        self.__width = 1 / self.n_classes / 2
        self.length = length
        print("object creation ended")

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        label = self.labels.iloc[idx]
        review_vec = None

        if isinstance(self.vectorizer, TfidfVectorizer):
            review_vec = self.vectorizer.transform([review]).toarray().reshape(-1, 1)
            if len(review_vec) == 0:
                review_vec = np.array([[0]])
            pad_item = [0]
        elif isinstance(self.vectorizer, WordVectors.GoogleW2V) or isinstance(self.vectorizer, WordVectors.GloveW2V):
            review_vec = self.vectorizer.convert_to_vec(review)
            if len(review_vec) == 0:
                review_vec = np.zeros((1, 300))
            pad_item = [0] * 300

        if len(review_vec) >= self.length:
            review_vec = review_vec[:self.length]
        else:
            diff = self.length - len(review_vec)
            padding = np.array([pad_item] * diff)
            review_vec = np.concatenate((review_vec, padding))

        if self.isReg:
            label = label / self.n_classes - self.__width
        else:
            label = label - 1

        label_vec = np.array([label])
        return review_vec.astype(np.float32), label_vec.astype(np.float32)


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

    rds = ReviewDataset(
        reviews_train["review"],
        reviews_train["rating"],
        "tfidf",
        isReg=False
    )

    dataloader = DataLoader(rds, 32)
    k = 0
    for batch_data, batch_label in dataloader:
        print(k + 1)
        k += 1
        print(batch_data.shape)
        print(batch_label.shape)
        print(batch_data)
        print()
        print(batch_label)
