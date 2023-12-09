import os
import pickle as pkl
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import config

MODEL_DIR = config.MODEL_DIR
os.makedirs(MODEL_DIR, exist_ok=True)


class DecisionTree:
    def __init__(self, tree_hyparams={'max_depth': 15}):
        self.vectorizer = TfidfVectorizer()
        self.model = DecisionTreeClassifier()
        self.model.set_params(**tree_hyparams)
        self.vec_path = os.path.join(MODEL_DIR, "dt_vectorizer.pkl")
        self.model_path = os.path.join(MODEL_DIR, "D_Tree.pkl")
        self.__trained = False        # checks if the current class object is trained or not

    def train_model(self, X, y):
        # if isinstance(X, pd.DataFrame)
        X_vec = self.vectorizer.fit_transform(X)
        self.model.fit(X_vec, y)
        self.__trained = True
        self.save_model()

    def save_model(self):
        with open(self.vec_path, "wb") as file:
            pkl.dump(self.vectorizer, file)

        with open(self.model_path, "wb") as file:
            pkl.dump(self.model, file)

    def __load_model(self):
        assert os.path.exists(self.vec_path)
        assert os.path.exists(self.model_path)

        with open(self.vec_path, "rb") as file:
            self.vectorizer = pkl.load(file)

        with open(self.model_path, "rb") as file:
            self.model = pkl.load(file)

    def predict(self, pred_x):
        if not self.__trained:
            self.__load_model()

        pred_x_vec = self.vectorizer.transform(pred_x)
        pred_y = self.model.predict(pred_x_vec)
        return pred_y
