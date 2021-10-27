from math import log, exp
from collections import defaultdict
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from selenium import webdriver
from bs4 import BeautifulSoup
import csv
from admin.common.models import ValueObject
import pandas as pd

# !pip install tensorflow-gpu==2.0.0-rc1
import tensorflow as tf
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


class GPUKoreanClassification(object):
    def __init__(self):
        pass

    def classify(self):
        ko_str = '이것은 한국어 문장입니다.'
        ja_str = 'これは日本語の文章です。'
        en_str = 'This is English Sentences.'
        x_train = [self.count_codePoint(ko_str),
                   self.count_codePoint(ja_str),
                   self.count_codePoint(en_str)]
        y_train = ['ko', 'ja', 'en']
        clf = GaussianNB()
        clf.fit(x_train, y_train)
        ko_test_str = '안녕하세요'
        ja_test_str = 'こんにちは'
        en_test_str = 'Hello'
        x_test = [self.count_codePoint(en_test_str),
                  self.count_codePoint(ja_test_str),
                  self.count_codePoint(ko_test_str)]
        y_test = ['en', 'ja', 'ko']
        y_pred = clf.predict(x_test)
        print(y_pred)
        print('정답률 : ', accuracy_score(y_test, y_pred))

    def count_codePoint(str):
        counter = np.zeros(65535)  # Unicode 코드 포인트 저장 배열
        for i in range(len(str)):
            code_point = ord(str[i])
            if code_point > 65535:
                continue
            counter[code_point] += 1

        counter = counter / len(str)
        return counter


class NaverMovie(object):
    def __init__(self):
        self.vo = ValueObject()
        self.vo.context = 'admin/nlp/data/'
        self.k = 0.5
        self.word_probs = []

    def naver_process(self):
        n = NaverMovie()
        n.model_fit()
        result = n.classify('내 인생 최고의 영화')
        print(f'result ::: {result}')
        result = n.classify('시간 아깝다. 정말 쓰레기다')
        print(f'result ::: {result}')
        result = n.classify('재미없어')
        print(f'result ::: {result}')
        # result::: 0.9634566316047457
        # result::: 0.00032621763187734896
        # result::: 0.01263560349465949

    def review_scraping(self):  # JH's
        ctx = self.vo.context
        driver = webdriver.Chrome(f'{ctx}chromedriver')
        driver.get('https://movie.naver.com/movie/point/af/list.naver?&page=1')
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        all_divs = soup.find_all('td', attrs={'class', 'title'})
        reviews = [str(td.br.next_element.string) for td in all_divs]
        for i, j in enumerate(reviews):
            reviews[i] = j.replace('\n', '')
            reviews[i] = reviews[i].replace('\t', '')
        # for i in reviews:
        #     i = i.replace()  X BECAUSE HERE i IS NOT AN INDEX LIKE ABOVE, THIS IS A KEY MISSING A VALUE
        ratings = [td.em.string for td in all_divs]
        result = {ratings[i]: reviews[i] for i in range(len(reviews))}
        with open(f'{ctx}naver_movie_review_dataset.csv', 'w', encoding='UTF-8', newline='') as f:
            wr = csv.writer(f, delimiter=',')
            wr.writerow(result.keys())
            wr.writerow(result.values())
        driver.close()

    # def web_scraping(self):  SIMILAR BUT NAH
    #     ctx = self.vo.context
    #     driver = webdriver.Chrome(f'{ctx}chromedriver')
    #     driver.get('https://movie.naver.com/movie/sdb/rank/rmovie.naver')
    #     soup = BeautifulSoup(driver.page_source, 'html.parser')
    #     all_divs = soup.find_all('div', attrs={'class', 'tit3'})
    #     products = [[div.a.string for div in all_divs]]
    #     with open(f'{ctx}naver_movie_dataset.csv', 'w', encoding='UTF-8', newline='') as f:
    #         # for product in products:
    #         # print(f'## {product}')
    #         wr = csv.writer(f)
    #         wr.writerows(products)  # 'products' is NOT LIST TYPE, but MATRIX TYPE
    #         # f.write(product)
    #     driver.get('https://movie.naver.com/movie/point/af/list.naver')
    #     all_divs = soup.find_all('div', attrs={'class', 'tit3'})
    #     products = [[div.a.string for div in all_divs]]
    #     with open(f'{ctx}review_train.csv', 'w', newline='', encoding='UTF-8') as f:
    #         wr = csv.writer(f)
    #         wr.writerows(products)
    #     driver.close()

    def load_corpus(self):
        # corpus = pd.read_table(f'{ctx}naver_movie_dataset.csv', sep=',', encoding='UTF-8')
        # corpus = pd.read_table(f'{ctx}review_train.csv', sep=',', encoding='UTF-8')
        # print(f'type(corpus) ::: {type(corpus)}')
        # print(f'corpus ::: {corpus}')
        corpus = pd.read_table(f'{self.vo.context}review_train.csv', sep=',', encoding='UTF-8')
        # print(f'type(corpus)::: {type(corpus)}')
        # print(f'corpus::: {corpus}')
        corpus = np.array(corpus)
        return corpus

    def count_words(self, train_X):
        # category 0 (+ve) 1 (-ve) => 'was fun': [1,0] // 'not fun': [0,1]
        counts = defaultdict(lambda: [0, 0])
        # initialises dict values, kinda sets dict format
        for doc, point in train_X:  # for loop *2 = matrix
            if self.isNumber(doc) is False:
                words = doc.split()
                for word in words:
                    counts[word][0 if point > 3.5 else 1] += 1
        return counts
        # print(f'word_counts ::: {dict(word_counts)}')

    def isNumber(self, s):
        try:
            float(s)
            return True  # if num then just yisi enough
        except ValueError:  # if not num then err
            return False  # goes back to ln.47 loop

    def word_probabilities(self, counts, n_class0, n_class1, k):
        return [(w,
                 (class0 + k) / (n_class0 + 2 * k),
                 (class1 + k) / (n_class1 + 2 * k))
                for w, (class0, class1) in counts.items()]

    def probability(self, word_probs, doc):
        docwords = doc.split()
        log_prob_if_class0 = log_prob_if_class1 = 0.0
        for word, prob_if_class0, prob_if_class1 in word_probs:
            if word in docwords:
                log_prob_if_class0 += log(prob_if_class0)
                log_prob_if_class1 += log(prob_if_class1)
            else:
                log_prob_if_class0 += log(1.0 - prob_if_class0)
                log_prob_if_class1 += log(1.0 - prob_if_class1)
        prob_if_class0 = exp(log_prob_if_class0)
        prob_if_class1 = exp(log_prob_if_class1)
        return prob_if_class0 / (prob_if_class0 + prob_if_class1)



    def model_fit(self):
        # ctx = self.vo.context
        # self.web_scraping()
        # train_X = np.array(corpus)
        train_X = self.load_corpus()

        num_class0 = len([1 for _, point in train_X if point > 3.5])
        num_class1 = len(train_X) - num_class0
        # [(i, j) for i, j in []]  list compre
        word_counts = self.count_words(train_X)
        self.word_probs = self.word_probabilities(word_counts, num_class0, num_class1, self.k)
        # print(type(ls))  # list
        # return ls

    def classify(self, doc):
        return self.probability(self.word_probs, doc)


class Imdb(object):
    def __init__(self):
        self.vo = ValueObject()
        self.vo.context = 'admin/nlp/data/'

    def decode_review(self, text, reverse_word_index):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])

    def imdb_process(self):
        imdb = keras.datasets.imdb
        (train_X, train_Y), (test_x, test_Y) = imdb.load_data(num_words=10000)
        print(f'>>>> {type(train_X)}')

        word_index = imdb.get_word_index()
        word_index = {k: (v + 3) for k, v in word_index.items()}
        word_index["<PAD>"] = 0
        word_index["<START>"] = 1
        word_index["<UNK>"] = 2  # unknown
        word_index["<UNUSED>"] = 3

        # reverse_word_index = dict([(v, k) for (k, v) in word_index.items()])
        # temp = self.decode_review(train_X[0], reverse_word_index)
        train_X = keras.preprocessing.sequence.pad_sequences(train_X,
                                                             value=word_index['<PAD>'],
                                                             padding='post',
                                                             maxlen=256)
        test_X = keras.preprocessing.sequence.pad_sequences(train_X,
                                                            value=word_index['<PAD>'],
                                                            padding='post',
                                                            maxlen=256)
        vocab_size = 10000
        model = keras.Sequential()
        model.add(keras.layers.Embedding(vocab_size, 16, input_shape=(None,)))
        model.add(keras.layers.GlobalAvgPool1D())
        model.add(keras.layers.Dense(16, activation=tf.nn.relu))
        model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
        model.compile(optimizer=tf.optimizers.Adam(), loss='binary_crossentropy', metrics=['acc'])
        x_val = train_X[:10000]
        partial_X_train = train_X[10000:]
        y_val = train_Y[:10000]
        partial_Y_train = train_Y[10000:]
        history = model.fit(partial_X_train, partial_Y_train, epochs=40,
                            batch_size=512, validation_data=(x_val, y_val))
        result = model.evaluate(test_X, test_Y)
        print(f'accuracy ::: {result}')
        # loss: 2.3566 - acc: 0.4965
        # [2.3565866947174072, 0.4965200126171112]

        history_dict = history.history
        history_dict.keys()
        acc = history_dict['acc']
        val_acc = history_dict['val_acc']
        # loss = history_dict['loss']
        # val_loss = history_dict['val_loss']
        epochs = range(1, len(acc) + 1)
        # "bo"는 "파란색 점"
        plt.plot(epochs, acc, 'bo', label='Training acc')
        # b는 "파란 실선"
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        # plt.show()
        # plt.clf()  # 그림을 초기화합니다
        plt.savefig(f'{self.vo.context}imdb_nlp3.png')
