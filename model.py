## Created by: Shubham Choudhury
## Created on: Sun 5 June 2022 8:43

"""
The function of this module is to use the pretrained sentiment analysis model and use product recommendation model
to recommend top 5 products
"""
##Importing libraries
import re
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
import contractions
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import pickle
stem = PorterStemmer()
lemma = WordNetLemmatizer()
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_column", None)




class ProductRecommendationEngine:
    ROOT_PATH = "pickle/"
    MODEL_NAME = "sentimentAnalysisModelLogisticRegression.pkl"
    VECTORIZER = "tfIDF.pkl"
    RECOMMENDER = "user_based_filtering.pkl"
    SCALER = "scaler.pkl"
    PCA = "pca.pkl"
    CLEAN_DATA = "data_export.pkl"
    THRESHOLD = 0.85

    def __init__(self):
        #Pull the sentiment prediction model
        self.model = pickle.load(open(ProductRecommendationEngine.ROOT_PATH + ProductRecommendationEngine.MODEL_NAME, 'rb'))
        #Pull the tfidf model
        self.vectorizer = pd.read_pickle(
            ProductRecommendationEngine.ROOT_PATH + ProductRecommendationEngine.VECTORIZER)
        #Pull the scaler model
        self.scaler = pd.read_pickle(
            ProductRecommendationEngine.ROOT_PATH + ProductRecommendationEngine.SCALER)
        # Pull the PCA model
        self.pca = pd.read_pickle(
            ProductRecommendationEngine.ROOT_PATH + ProductRecommendationEngine.PCA)
        #Pull the cleaned up data
        self.data = pd.read_pickle(
            ProductRecommendationEngine.ROOT_PATH + ProductRecommendationEngine.CLEAN_DATA)
        #Pull the recommendation model
        self.recommender = pd.read_pickle(
            ProductRecommendationEngine.ROOT_PATH + ProductRecommendationEngine.RECOMMENDER)
        self.stopwords = set(stopwords.words('english'))
        self.THRESHOLD = ProductRecommendationEngine.THRESHOLD
        pass

    def get_postags(self, word):

        """
        Function: Map POS tag to first character lemmatize() accepts
        :param word:
        :return:
        """
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    def normalize(self, sentence):
        """
        Function: Correcting words that contains extra alphabets incorrectly
        :param sentence:
        :return:
        """
        from textblob import TextBlob
        from textblob import Word
        rx = re.compile(r'([^\W\d_])\1{2,}')
        return re.sub(r'[^\W\d_]+',
                      lambda x: Word(rx.sub(r'\1\1', x.group())).correct() if rx.search(x.group()) else x.group(),
                      sentence)


    def preprocess_text(self, data):
        """
        Function: Pre-processing the text to be used for vectorzation
        :param data:
        :return:
        """
        data['reviews_length'] = data['reviews_text'].apply(lambda x: len(str.split(x, " ")))
        data['reviews_text'] = data['reviews_text'].apply(lambda x: " ".join(
            [contractions.fix(i) for i in str.split(x)]))
        data['Feature'] = data.apply(lambda x: x['reviews_text'] + " " +
                                               x['reviews_title'], axis=1)
        pattern = re.compile("[0-9]+|[$|\.]|www\.\S+")
        pattern_2 = re.compile("|[^\w\s]")
        data['tokenized_reviews'] = data['Feature'].apply(lambda x: " ".join([str.lower(lemma.lemmatize(i, self.get_postags(i)))
                                                                              for i in nltk.word_tokenize(re.sub(pattern_2, "",
                                                                                                                 re.sub(pattern, "", x)))]))
        data['tokenized_reviews'] = data['tokenized_reviews'].apply(lambda x: self.normalize(x))
        print("Text Preprocessed successfully")
        return data

    def feature_extraction(self, data):
        """
        Function: Applying TF_idf and normalization into the data set and create features
        :param data:
        :return:
        """
        tfv = self.vectorizer
        l = list(data['tokenized_reviews'])
        b = tfv.transform(l)
        df = pd.DataFrame(b.toarray(), columns=tfv.get_feature_names_out())
        df.reset_index(drop=True, inplace=True)
        data.reset_index(drop = True, inplace = True)
        data = pd.concat([data[['reviews_length']], df], axis=1)
        scaler = self.scaler
        data = scaler.transform(data)
        pca = self.pca
        data = pca.transform(data)
        print("Features Extracted Successfully")
        return data

    def predict_sentiment(self, review_text, review_title):
        THRESHOLD = self.THRESHOLD
        """
        Function: Sentiment prediction of any review_text
        :param review_text:
        :param review_title:
        :return:
        """
        dict = {"reviews_text": str(review_text),
                "reviews_title": str(review_title)}
        data = pd.DataFrame(dict, index=[0])
        data = self.preprocess_text(data)
        data = self.feature_extraction(data)
        y_pred = np.where(self.model.predict_proba(data)[:,1] > THRESHOLD, 1, 0)
        y_pred_proba = self.model.predict_proba(data)
        return [int(y_pred), y_pred_proba[0][1]]


    def product_recommendations(self, user):
        THRESHOLD = self.THRESHOLD

        """
        Function: Main function for product recommendation

        :param user:
        :return:
        """
        if (user in self.recommender.index):
            recommendations = list(self.recommender.loc[user].sort_values(ascending=False)[0:20].index)
            temp = self.data[self.data.id.isin(recommendations)]
            data = self.preprocess_text(temp)
            X = self.feature_extraction(data)
            data["predicted_sentiment"] = np.where(self.model.predict_proba(X)[:,1] > THRESHOLD, 1, 0)
            df = data[['id', 'predicted_sentiment']]
            data_grouped = df.groupby('id', as_index=False).count()
            data_grouped["pos_review_count"] = data_grouped.id.apply(
                lambda x: df[(df.id == x) & (df.predicted_sentiment == 1)]["predicted_sentiment"].count())
            data_grouped["total_review_count"] = data_grouped['predicted_sentiment']
            data_grouped['pos_sentiment_percent'] = np.round(
                data_grouped["pos_review_count"] / data_grouped["total_review_count"] * 100, 2)
            data_grouped = data_grouped.sort_values(
                'pos_sentiment_percent', ascending=False)[0:5]
            return pd.merge(self.data, data_grouped, on="id")[["name", "brand", "pos_sentiment_percent"]].drop_duplicates().sort_values(['pos_sentiment_percent', 'name'], ascending=[False, True])
        else:
            print("User doesn't exist. Please try again later")

# if __name__ == "__main__":
#     x = ProductRecommendationEngine()
#     d = x.predict_sentiment("Awesome product", "Really Loved it")
#     print(d)