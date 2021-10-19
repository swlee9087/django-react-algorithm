import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from admin.common.models import ValueObject


class FashionClassification(object):

    def __init__(self):
        self.vo = ValueObject()
        self.vo.context = 'admin/tensor/data/'
        self.class_name = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def fashion(self):  # django
        self.get_data()

    def hook(self):  # 전체 다 걸림
        images = self.get_data()
        model = self.create_model()
        model = self.train_model()
        arr = self.predict()
        self.plot_image()
        self.plot_value_array()

    def get_data(self) -> []:
        fashion_mnist = keras.datasets.fashion_mnist
        (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
        print(X_train_full.shape)
        print(X_train_full.dtype)
        print(f'train 행 : {X_train_full.shape[0]} 열 : {X_train_full.shape[1]}')
        print(f'test 행 : {X_test.shape[0]} 열 : {X_test.shape[1]}')
        plt.figure()
        plt.imshow(X_train_full[3])
        plt.colorbar()
        plt.grid(False)
        plt.savefig(f'{self.vo.context}fashion_random.png')
        return [X_train_full, y_train_full, X_test, y_test]

    def preprocess(self):
        pass

    def create_model(self) -> object:
        pass

    def train_model(self) -> object:
        pass

    def test_model(self) -> object:
        pass

    def predict(self):
        pass

    def plot_image(self):
        pass

    def plot_value_array(self):
        pass


class AdalineGD(object):  # 적응형 선형 뉴런 분류기.  GD = gradient descent 경사 하강법. GD = atom???

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        # X : {array-like}, shape = [n_samples, n_features]
        #           n_samples 개의 샘플과 n_features 개의 특성으로 이루어진 훈련 데이터

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []  # 에포크마다 누적된 비용 함수의 제곱합

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            # Please note that the "activation" method has no effect
            # in the code since it is simply an identity function. We
            # could write `output = self.net_input(X)` directly instead.
            # The purpose of the activation is more conceptual, i.e.,
            # in the case of logistic regression (as we will see later),
            # we could change it to
            # a sigmoid function to implement a logistic regression classifier.
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0  # bias 제곱하는 activation fn.
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return X

    def predict(self, X):  # 단위 계단 함수를 사용하여 클래스 레이블을 반환
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        #   매개변수
        #   eta : float 학습률 (0.0~1.0)
        #   n_iter : int 훈련 데섹 반복 횟수
        #   random_state : int weight 무작위 초기화 위한 난수 생성기
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y) -> object:  # 훈련 데터 학습  -> matrix
        #   속성
        #   w_ : 1d-array : 학습된 weight
        #   errors_ : list 에포크마다 누적된 분류 오류
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def predict(self, X):  # unit step fn으로 클래스 레블 반환
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def net_input(self, X):  # 최종 입력 계산
        return np.dot(X, self.w_[1:]) + self.w_[0]


class Calculator(object):

    def __init__(self):
        print(f'Tensorflow Version: {tf.__version__}')

    def process(self):
        self.plus(4, 8)
        print('*' * 80)
        self.mean()
        print('*' * 80)

    def plus(self, a, b):
        print(tf.constant(a) + tf.constant(b))

    def minus(self, a, b):
        print(tf.constant(a) - tf.constant(b))

    def multiply(self, a, b):
        print(tf.constant(a) * tf.constant(b))

    def divide(self, a, b):
        print(tf.constant(a) / tf.constant(b))

    def mean(self):
        x_array = np.arange(18).reshape(3, 2, 3)
        x2 = tf.reshape(x_array, shape=(-1, 6))
        # 각 열의 합
        xsum = tf.reduce_sum(x2, axis=0)
        # 각 열의 평균
        xmean = tf.reduce_mean(x2, axis=0)

        print(f'입력 크기: {x_array.shape} \n')
        print(f'크기가 변경된 입력 크기: {x2.numpy()} \n')
        print(f'열의 합: {xsum.numpy()} \n')
        print(f'열 평균: {xmean.numpy()} \n')
