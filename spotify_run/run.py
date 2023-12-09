import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

import config
from preprocessing import EngPreprocessing
from NaiveBayes import NaiveBayes
from DecisionTree import DecisionTree
from BiLSTM import BiLSTM_Classifier
import metrics

DATASET_PATH = config.DATASET_PATH

reviews_df = pd.read_csv(DATASET_PATH)
# making the column names lowercase
reviews_df.columns = [colname.lower() for colname in reviews_df.columns]

# splitting the data
reviews_train, reviews_test = train_test_split(reviews_df,
                                               train_size=0.8,
                                               random_state=711
                                               )

# -------- data cleaning for train data -----------------
eng_preproc = EngPreprocessing()
reviews_train["reviews_processed"] = reviews_train["review"].apply(eng_preproc.preprocess)
# removing the records with reviews_processed length == 0
# print(f"length before masking: {len(reviews_train)}")
mask = reviews_train.reviews_processed.str.len() > 0
reviews_train = reviews_train[mask]
# print(f"length after masking: {len(reviews_train)}")

# ----------- data cleaning for test data ----------------
reviews_test["reviews_processed"] = reviews_test["review"].apply(eng_preproc.preprocess)
mask = reviews_test.reviews_processed.str.len() > 0
reviews_test = reviews_test[mask]

# -------------- training the naivebayes CountVectorizer -------------------
nb_cntvec = NaiveBayes(CountVectorizer())
nb_cntvec.train_model(reviews_train["reviews_processed"], reviews_train["rating"])

y_pred = nb_cntvec.predict(reviews_test.reviews_processed)
print(f"Test Accuracy with NaiveBayes (CountVectorizer): {metrics.accuracy(y_pred, reviews_test.rating)}")

# -------------- training the naivebayes tfidf-------------------
nb_tfidf = NaiveBayes(TfidfVectorizer())
nb_tfidf.train_model(reviews_train["reviews_processed"], reviews_train["rating"])

y_pred = nb_tfidf.predict(reviews_test.reviews_processed)
print(f"Test Accuracy with NaiveBayes (TfIdf): {metrics.accuracy(y_pred, reviews_test.rating)}")

# -------------- training the Decision Tree tfidf-------------------
dt_tfidf = DecisionTree()
dt_tfidf.train_model(reviews_train["reviews_processed"], reviews_train["rating"])

y_pred = dt_tfidf.predict(reviews_test.reviews_processed)
print(f"Test Accuracy with Decision Tree (TfIdf): {metrics.accuracy(y_pred, reviews_test.rating)}")
conf_dt = metrics.confusion_matrix_(y_pred, reviews_test.rating)
for key in conf_dt:
    print(f"confusion matrix {key}")
    print(conf_dt[key])
    print()

# ------------------- training with BiLSTM --------------------------
b_clf = BiLSTM_Classifier(np.unique(reviews_train["rating"]), "tfidf", 64)
b_clf.trainBiLSTM(
    reviews_train["reviews"],
    reviews_train["rating"],
    epochs=3,
    optimizer="adam",
    batch_size=4
)
