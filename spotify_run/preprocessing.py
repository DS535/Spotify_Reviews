import re
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import emoji
import unicodedata


class EngPreprocessing:
    def __init__(self):
        self.pipeline = [
            cannonical_decomposition,
            str.lower,
            process_emojis,
            remove_non_eng_char_and_nums,
            whitespaces_to_space,
            stemming,
            stopword_removal
        ]

    def preprocess(self, text):
        for preproc_func in self.pipeline:
            text = preproc_func(text)

        return text


def remove_non_eng_char_and_nums(text):
    """
    Removes all non-english characters and numbers from the text
    :param text: (str) A string of charecters
    :return: (str) removing all non-english alphanum characters
    """
    eng_chars = []
    for t_char in text:
        if ('a' <= t_char <= 'z') or ('0' <= t_char <= '9') or (re.match(r"\s", t_char)):
            eng_chars.append(t_char)
        else:
            eng_chars.append(t_char)
    return "".join(eng_chars)


def whitespaces_to_space(text):
    """
    converts all whitespaces and new lines by single space
    :param text: (str) A string to be processed
    :return: (str) processed string
    """
    text = re.sub(r"\s+", " ", text)
    return text


def stemming(text, language="english"):
    snb = SnowballStemmer(language)
    stemmed_words = []
    for word in text.split():
        stemmed_words.append(snb.stem(word))

    return " ".join(stemmed_words)


def stopword_removal(text, language="english"):
    wo_stopwords = []
    for word in text.split():
        if word not in stopwords.words(language):
            wo_stopwords.append(word)

    return " ".join(wo_stopwords)


def process_emojis(text, language="english"):
    lang = "en"
    if language.lower() == "english":
        lang = "en"
    text_ = emoji.demojize(text, language=lang)
    text_ = text_.replace("_", " ")
    return text_


def cannonical_decomposition(text):
    # this function helps to remove get english written in different fonts
    # like "𝗧𝗼 𝗯𝗲 𝗵𝗼𝗻𝗲𝘀𝘁 𝗜 𝗿𝗲𝗮𝗹𝗹𝘆 𝗹𝗶𝗸𝗲𝗱 𝘁𝗵𝗲 𝗮𝗽𝗽"
    try:
        text_ = unicodedata.normalize("NFKC", text)
    except:
        text_ = text

    return text_

