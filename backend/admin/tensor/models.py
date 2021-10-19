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
        # self.get_data()
        self.hook()

    def hook(self):  # 전체 다 걸림
        # images = self.get_data()
        [train_images, train_labels, test_images, test_labels] = self.get_data()
        model = self.create_model()
        # model = self.train_model(model, X_train_full, y_train_full)
        # model = self.train_model(model, ls[0], ls[1])  NOPE
        # self.test_model(model)
        # arr = self.predict(model, ls[2], ls[3], 0)  # leave index val as 0
        # prediction = arr[0]
        # test_images = arr[1]
        # test_labels = arr[2]
        model.fit(train_images, train_labels, epochs=5)
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
        # verbose 는 학습하는 내부상황 보기 중 2번선택
        predictions = model.predict(test_images)
        i = 5  # change for diff pic
        # print(f'prediction: {prediction}')
        # print(f'test img: {prediction}')
        # print(f'test val: {prediction}')
        # pred = predictions[i]
        # answer = test_labels[i]
        print(f'모델이 예측한 값 {np.argmax(predictions)}')
        print(f'정답: {test_labels[i]}')
        print(f'테스트 정확도: {test_acc}')
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)  # row 1 col 2 position 1st
        # prediction_array, true_label, image = predictions[i], test_labels[i], test_images[i]
        test_image, test_predictions, test_label = test_images[i], predictions[i], test_labels[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(test_image, cmap=plt.cm.binary)
        test_pred = np.argmax(test_predictions)
        print('f{test_pred}')
        print('#' * 100)
        print('f{test_label}')
        # if test_pred == test_label:
        #     color = 'blue'
        # else:
        #     color = 'red'
        # plt.xlabel('{} : {}%'.format(self.class_names[test_pred],
        #                              100 * np.max(test_predictions),
        #                              self.class_names[test_label], color))
        plt.subplot(1,2,2)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        this_plot = plt.bar(range(10), test_pred, color='#777777')
        plt.ylim([0,1])
        test_pred = np.argmax(test_predictions)
        this_plot[test_pred].set_color('red')
        this_plot[test_label].set_color('blue')
        plt.savefig(f'{self.vo.context}fashion_answer.png')

    def get_data(self) -> []:
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()
        # self.peek_datas(X_train_full, X_test, y_train_full)
        # return [X_train_full, y_train_full, X_test, y_test]  # ls base
        return [train_images, train_labels, test_images, test_labels]

    def peek_datas(self, train_images, test_images, train_labels):
        print(train_images.shape)
        print(train_images.dtype)
        print(f'train 행 : {train_images.shape[0]} 열 : {train_images.shape[1]}')
        print(f'test 행 : {test_images.shape[0]} 열 : {test_images.shape[1]}')
        plt.figure()
        plt.imshow(train_images[3])
        plt.colorbar()
        plt.grid(False)
        plt.savefig(f'{self.vo.context}fashion_random.png')

        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[i], cmap=plt.cm.binary)
            plt.xlabel(self.class_name[train_labels[i]])
        plt.savefig(f'{self.vo.context}fashion_subplot.png')

    def create_model(self) -> object:
        # model = keras.models.Sequential()  # "sequential model"
        # model.add(keras.layers.Flatten(input_shape=[28, 28]))  # input(matrix) layer flat.
        # # also sets pixel size for fshrdm.png
        # model.add(keras.layers.Dense(300, activation="relu"))  # neuron count 300
        # # model.add(keras.layers.Dense(100, activation="relu"))  # he thinks it's duplicate
        # model.add(keras.layers.Dense(10, activation="softmax"))  # output layer activation fn
        model = keras.Sequential([  # TF book p373
            keras.layers.Flatten(input_shape=[28, 28]),
            # keras.layers.Dense(300, activation="relu"),
            keras.layers.Dense(128, activation="relu"),  # 300 -> 128
            keras.layers.Dense(10, activation="softmax")
        ])
        # model.summary()  # Tf book p375
        # model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    # def train_model(self, model, train_images, train_labels) -> object:
    #     model.fit(train_images, train_labels, epoch=5)
    #     return model
    #
    # def test_model(self, model, test_images, test_labels) -> object:
    #     test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    #     print(f'test accuracy: {test_acc}')
    #
    # def predict(self, model, test_images, test_labels, index):
    #     prediction = model.predict(test_images)
    #     pred = prediction[index]
    #     answer = test_labels[index]
    #     print(f' model predicted value {np.argmax(pred)}')
    #     print(f'answer: {answer}')
    #     return [prediction, test_images, test_labels]
    #
    # def plot_image(self):
    #     pass
    #
    # def plot_value_array(self):
    #     pass


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
        self.cost_ = []  #  accum 비용 함수의 제곱 sum per epoch

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


class Perceptron(object):  # perceptron classifier
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
