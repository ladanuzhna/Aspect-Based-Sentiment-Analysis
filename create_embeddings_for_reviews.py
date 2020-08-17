import pandas as pd
import en_core_web_sm
from nltk import word_tokenize
from nltk.corpus import stopwords
from unidecode import unidecode
import string
from gensim.models import KeyedVectors
from wikipedia2vec import Wikipedia2Vec
import neuralcoref
import numpy as np


def clean(text):
    # convert text to lower case
    text = text.lower()
    stopset = stopwords.words('english') + list(string.punctuation)
    # remove stop words and punctuations
    # word_tokenize is used to tokenize the input corpus in word tokens.
    text = " ".join([i for i in word_tokenize(text) if i not in stopset])
    return unidecode(text)


def relevant(noun, feature_vectors, similarity_threshold=0.5):
    """
    Convert noun to word2vec embedding
    Check if there are similar items in given feature vectors
    If yes, return True
    """
    vector = model.wv[noun]
    word_distances = []
    for word in feature_vectors:
        word_distances.append(model.wv.similarity(word, vector))
    if max(word_distances) > similarity_threshold:
        return True
    return False


def extract_aspects(text, features):
    """
    text is a sentence from a review
    features is a list of word2vec embeddings of features you are searching for
    """
    irrelevant_aspects = []
    relevant_aspects = []

    doc = nlp(clean(text))
    doc_index = 0

    for tok in doc:
        if (tok.pos_ == 'ADJ' and doc[doc_index - 1].pos_ == 'NOUN'):
            # CHECK HERE IF YOUR ASPECT IS SIMILAR TO ANY OF THE ASPECTS WE ARE SEARCHING FOR~~~
            if relevant(doc[doc_index - 1], features):
                relevant_aspects.append(str(doc[doc_index - 1]) + " " + str(tok))
            else:
                irrelevant_aspects.append(str(doc[doc_index - 1]) + " " + str(tok))
        doc_index = doc_index + 1

    return irrelevant_aspects, relevant_aspects


if __name__ == "__main__":
    # Load pretrained word2vec
    model = KeyedVectors.load_word2vec_format('enwiki_20180420_nolg_100d.txt')
    nlp = en_core_web_sm.load(parse=True, tag=True, entity=False)
    df_bestbuy = pd.read_csv('bestbuy_dataset.csv', error_bad_lines=False)[100:110]

    # Pass your reviews through it
    # for review in df_bestbuy['Reviews']:
    #     if review == review:
    #         review = resolve_coreferences(review)
    #         for sentence in review.split('.'):
    #             model.train([clean(sentence)], total_examples=1, epochs=1)

    #Create embeddings for reviews
    encoded_df = df_bestbuy[['Reviews','Model','Name']].copy()
    encoded_reviews = []
    cleaned_reviews = []
    for i,review in enumerate(df_bestbuy['Reviews']):
        if review == review:
            # create embedding and appended it to encoded array
            # also append cleaned review
            review = clean(review)
            cleaned_reviews.append([sentence for sentence in review.split('.')])
            rev_embedding = []

            for sentence in review.split('.'):
                sent_embedding = []
                for word in sentence.split():
                    sent_embedding.append(model[word])

                rev_embedding.append(sent_embedding)
            encoded_reviews.append(rev_embedding)
        else:
            # there is no review -> append none
            encoded_reviews.append(None)
            cleaned_reviews.append(None)

        encoded_df['Clean review'] = pd.Series(cleaned_reviews)
        encoded_df['Encoded review'] = pd.Series(encoded_reviews)
        encoded_df.to_csv('encoded_df.csv')
