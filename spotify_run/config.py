import torch

DATASET_PATH = "../dataset/reviews.csv"
MODEL_DIR = r"D:\MTech\sem3\NLU\assignments\assignment1\models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Glove_path = r"D:\MTech\sem3\NLU\assignments\assignment1\embeddings\glove.6B"
GLOVE_50 = r"D:\MTech\sem3\NLU\assignments\assignment1\embeddings\glove.6B\glove.6B.50d.txt"
GLOVE_100 = r"D:\MTech\sem3\NLU\assignments\assignment1\embeddings\glove.6B\glove.6B.100d.txt"
GLOVE_200 = r"D:\MTech\sem3\NLU\assignments\assignment1\embeddings\glove.6B\glove.6B.200d.txt"
GLOVE_300 = r"D:\MTech\sem3\NLU\assignments\assignment1\embeddings\glove.6B\glove.6B.300d.txt"
