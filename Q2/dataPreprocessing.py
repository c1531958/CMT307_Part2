import operator

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_selection import chi2, SelectKBest

import numpy as np


lemmatizer = nltk.stem.WordNetLemmatizer()
english_stop_words = stopwords.words('english')
# remove these stop words to find negated bigrams and verb grams, eg "not like", "I loved"
english_stop_words.remove('not')
english_stop_words.remove('i')
# from html </br>
english_stop_words.append('br')
punct =['.',';',':','!','\'', '?', '"', '(', ')', '[', ']', '<', '>', '\\', '/']


def rm_punct(list_tokens):
    no_punct = [i for i in list_tokens if i not in punct]
    return no_punct

# Function taken from Session 1


def get_list_tokens(string):
    sentence_split = nltk.tokenize.sent_tokenize(string)
    list_tokens = []
    for sentence in sentence_split:
        list_tokens_sentence = nltk.tokenize.word_tokenize(sentence)
        for token in list_tokens_sentence:
            list_tokens.append(lemmatizer.lemmatize(token).lower())

    return list_tokens


def remove_stop_words(list_tokens):
    clean_list_tokens = []
    for token in list_tokens:
        if token not in english_stop_words:
            clean_list_tokens.append(token)

    return clean_list_tokens


def getAdjectives(list_tokens_sentence):
    adjectives = [word for (word, pos) in pos_tag(
        list_tokens_sentence, tagset='universal') if pos == 'ADJ' and len(word) > 2]
    return adjectives


def getNegatedBigrams(rm_st):
    bigrams = list(nltk.bigrams(rm_st))
    negation_bigrams = []
    for (word_1, word_2) in bigrams:
        if word_1 in ['not', 'n\'t']:
            for (word, pos) in pos_tag([word_2], tagset='universal'):
                if pos == 'ADJ':
                    negation_bigrams.append(('not', word))

    return negation_bigrams


def getVerbGrams(rm_st):
    trigrams = list(nltk.trigrams(rm_st))
    verb_grams = []
    for (word_1, word_2, word_3) in trigrams:
        if word_1 == 'i':
            for (word, pos) in pos_tag([word_2, word_3], tagset='universal'):
                # to escape "'m", "'ve" "'d" etc
                if not word.startswith('\''):
                    if pos == 'VERB':
                        # normalise werb seen-->see
                        word = WordNetLemmatizer().lemmatize(word, 'v')
                        if word_2 in ['not', 'n\'t']:
                            verb_grams.append(('not', word))
                        else:
                            verb_grams.append(word)

    return verb_grams


def getVocabulary(training_set):
    dict_adj_frequency = {}
    total_negation_bigrams_fr = {}
    total_verb_grams_fr = {}
    vocabulary = []

    for review in training_set:
        sentence_tokens = get_list_tokens(review)
        rm_st = remove_stop_words(sentence_tokens)
        adjectives = getAdjectives(rm_punct(rm_st))
        negation_bigrams = getNegatedBigrams(rm_st)
        verb_grams = getVerbGrams(rm_st)

        for word in adjectives:
            if word not in dict_adj_frequency:
                dict_adj_frequency[word] = 1
            else:
                dict_adj_frequency[word] += 1

        for t in negation_bigrams:
            if t not in total_negation_bigrams_fr:
                total_negation_bigrams_fr[t] = 1
            else:
                total_negation_bigrams_fr[t] += 1

        for t in verb_grams:
            if t not in total_verb_grams_fr:
                total_verb_grams_fr[t] = 1
            else:
                total_verb_grams_fr[t] += 1

    sorted_adj = sorted(dict_adj_frequency.items(),
                        key=operator.itemgetter(1), reverse=True)[:1000]
    sorted_bigrams = sorted(total_negation_bigrams_fr.items(
    ), key=operator.itemgetter(1), reverse=True)[:1000]
    sorted_ngrams = sorted(total_verb_grams_fr.items(),
                           key=operator.itemgetter(1), reverse=True)[:200]

    for word, frequency in sorted_adj:
        vocabulary.append(word)
    for word, frequency in sorted_bigrams:
        vocabulary.append(word)
    for word, frequency in sorted_ngrams:
        vocabulary.append(word)

    return vocabulary


def getVectorText(list_vocab, string):
    vector_text = np.zeros(len(list_vocab))
    sentence_tokens = get_list_tokens(string)
    rm_st = remove_stop_words(sentence_tokens)
    adjectives = getAdjectives(rm_punct(rm_st))
    negation_bigrams = getNegatedBigrams(rm_st)
    verb_grams = getVerbGrams(rm_st)
    for i, word in enumerate(list_vocab):
        if word in adjectives:
            vector_text[i] = adjectives.count(word)
        elif word in negation_bigrams:
            vector_text[i] = negation_bigrams.count(word)
        elif word in verb_grams:
            vector_text[i] = verb_grams.count(word)
    return vector_text


def getXY(df_train_pos, df_train_neg, vocabulary):
    X_train = []
    Y_train = []
    counter = 0
    for pos_review in df_train_pos:
        vector_pos_review = getVectorText(vocabulary, pos_review)
        X_train.append(vector_pos_review)
        Y_train.append(1)

    for neg_review in df_train_neg:
        vector_neg_review = getVectorText(vocabulary, neg_review)
        X_train.append(vector_neg_review)
        Y_train.append(0)

    return np.asarray(X_train), np.asarray(Y_train)


# feature selection based on chi2
def selectBestFeatures(X_train, Y_train, feature_count):
    X_train_sentanalysis = np.asarray(X_train)
    Y_train_sentanalysis = np.asarray(Y_train)

    # feature selection with chi-squared test
    fs_sentanalysis = SelectKBest(chi2, k=feature_count).fit(
        X_train_sentanalysis, Y_train_sentanalysis)
    X_train_new = fs_sentanalysis.transform(X_train_sentanalysis)
    features_selected = fs_sentanalysis.get_support()

    return X_train_new, features_selected
