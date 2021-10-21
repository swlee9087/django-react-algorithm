from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf
import os

from admin.common.models import ValueObject, Reader


class Iris(object):
    def __init__(self):
        self.vo = ValueObject()
        self.vo.context = 'admin/iris/data/'

    # def hook(self):  # too much then write hook
    #     self.base()

    def iris_by_tf(self):
        reader = Reader()
        vo = self.vo
        train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"

        train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                                   origin=train_dataset_url)
        test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"

        test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                          origin=test_url)

        # print(f"Local copy of the dataset file: {train_dataset_fp}")
        # print(f"Local copy of the dataset file: {test_fp}")
        # download first then get download position
        # print(f'Iris data top 5: {train_dataset_fp.head(5)}')
        vo.fname = 'iris_training'
        iris_df = reader.csv(reader.new_file(vo))
        # print(f'iris_df HEAD: {iris_df.head(3)}')

        # column order in CSV file
        column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

        feature_names = column_names[:-1]
        label_name = column_names[-1]

        # print(f"Features: {feature_names}")
        # print(f"Label: {label_name}")

        class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
        batch_size = 32

        train_dataset = tf.data.experimental.make_csv_dataset(
            train_dataset_fp,
            batch_size,
            column_names=column_names,
            label_name=label_name,
            num_epochs=1)
        features, labels = next(iter(train_dataset))

        # print(features)
        plt.scatter(features['petal_length'],
                    features['sepal_length'],
                    c=labels,
                    cmap='viridis')

        plt.xlabel("Petal length")
        plt.ylabel("Sepal length")
        # plt.savefig(f'{self.vo.context}iris_tf_scatter.png')

        test_dataset = tf.data.experimental.make_csv_dataset(
            test_fp,
            batch_size,
            column_names=column_names,
            label_name='species',
            num_epochs=1,
            shuffle=False)

        train_dataset = train_dataset.map(self.pack_features_vector)
        features, labels = next(iter(train_dataset))
        print(f'features[:5] value: {features[:5]}')
        test_dataset = test_dataset.map(self.pack_features_vector)
        features, labels = next(iter(test_dataset))
        print(f'test_dataset features[:5] 의 값 : {features[:5]}')
        # model is the relation btwn feature and label.
        # 좋은 머신러닝 접근 방식이라면 적절한 모델을 제시해 줍니다.
        # 적절한 머신러닝 모델 형식에 충분한 대표 예제를 제공하면 프로그램이 관계를 파악해 줍니다.
        # There are several categories of neural networks
        # and this program uses a dense, or fully-connected (tf.nn) neural network.
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
            tf.keras.layers.Dense(10, activation=tf.nn.relu),
            tf.keras.layers.Dense(3)
        ])
        # There are many tf.keras.activations, but ReLU is common for hidden layers.
        predictions = model(features)
        # predictions[:5]
        print(f'predictions[:5] value: {predictions[:5]}')
        print(f'tf.nn.softmax(predictions[:5] value: {tf.nn.softmax(predictions[:5])}')
        print(f'tf.argmax(predictions, axis=1) value: {tf.argmax(predictions, axis=1)}')
        print(f'Labels value: {labels}')

        # If you learn too much about the training dataset,
        # then the predictions only work for the data it has seen and will not be generalizable.
        # This problem is called overfitting, like memorizing the answers
        # instead of understanding how to solve a problem.
        # Our model will calculate its loss using the
        # tf.keras.losses.SparseCategoricalCrossentropy function
        # which takes the model's class probability predictions and the desired label,
        # and returns the average loss across the examples.

        l = self.loss(model, features, labels, training=False)
        print(f"Loss test: {l}")
        # An optimizer applies the computed gradients
        # to the model's variables to minimize the loss function.
        # Gradually, the model will find the best combination of weights and bias to minimize loss.
        # And the lower the loss, the better the model's predictions.
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        loss_value, grads = self.grad(model, features, labels)

        print(f"Step: {optimizer.iterations.numpy()}, Initial Loss: {loss_value.numpy()}")

        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        print(f"Step: {optimizer.iterations.numpy()},         Loss: {self.loss(model, features, labels, training=True).numpy()}")
        # An epoch is one pass through the dataset.
        # Use an optimizer to update the model's variables.
        # num_epochs variable is the number of times to loop over the dataset collection.

        ## Note: Rerunning this cell uses the same model variables

        # Keep results for plotting
        train_loss_results = []
        train_accuracy_results = []

        num_epochs = 201

        for epoch in range(num_epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

            # Training loop - using batches of 32
            for x, y in train_dataset:
                # Optimize the model
                loss_value, grads = self.grad(model, x, y)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                # Track progress
                epoch_loss_avg.update_state(loss_value)  # Add current batch loss
                # Compare predicted label to actual label
                # training=True is needed only if there are layers with different
                # behavior during training versus inference (e.g. Dropout).
                epoch_accuracy.update_state(y, model(x, training=True))

            # End epoch
            train_loss_results.append(epoch_loss_avg.result())
            train_accuracy_results.append(epoch_accuracy.result())

            if epoch % 50 == 0:
                print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                            epoch_loss_avg.result(),
                                                                            epoch_accuracy.result()))
        fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
        fig.suptitle('Training Metrics')

        axes[0].set_ylabel("Loss", fontsize=14)
        axes[0].plot(train_loss_results)

        axes[1].set_ylabel("Accuracy", fontsize=14)
        axes[1].set_xlabel("Epoch", fontsize=14)
        axes[1].plot(train_accuracy_results)
        # plt.savefig(f'{self.vo.context}train_accuracy_results.png')

        # Unlike the training stage, the model only evaluates a single epoch of the test data.
        test_accuracy = tf.keras.metrics.Accuracy()

        for (x, y) in test_dataset:
            # training=False is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            logits = model(x, training=False)
            prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
            test_accuracy(prediction, y)

        print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
        print(f'tf.stack([y,prediction],axis=1) result: {tf.stack([y,prediction],axis=1)}')

        # predict with trained model
        predict_dataset = tf.convert_to_tensor([
            [5.1, 3.3, 1.7, 0.5, ],
            [5.9, 3.0, 4.2, 1.5, ],
            [6.9, 3.1, 5.4, 2.1]
        ])

        # training=False is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(predict_dataset, training=False)

        for i, logits in enumerate(predictions):
            class_idx = tf.argmax(logits).numpy()
            p = tf.nn.softmax(logits)[class_idx]
            name = class_names[class_idx]
            print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100 * p))

    def grad(self, model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss(model, inputs, targets, training=True)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def loss(self, model, x, y, training):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # training=training is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        y_ = model(x, training=training)

        return loss_object(y_true=y, y_pred=y_)

    def pack_features_vector(self, features, labels):
        """Pack the features into a single array."""
        features = tf.stack(list(features.values()), axis=1)
        return features, labels

    def base(self):
        np.random.seed(0)
        iris = load_iris()
        iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
        # print(f'Iris Data Structure: {iris_df.head(2)} \n {iris_df.columns}')
        # ['sepal length (cm)', 'sepal width (cm)',
        # 'petal length (cm)', 'petal width (cm)']

        iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        # print(f'(Species Added) Iris Data Structure: {iris_df.head(2)} \n {iris_df.columns}')
        # ['sepal length (cm)', 'sepal width (cm)',
        # 'petal length (cm)', 'petal width (cm)',
        # 'species']

        iris_df['is_train'] = np.random.uniform(0, 1, len(iris_df)) <= 0.75  # train set 75%
        train, test = iris_df[iris_df['is_train'] == True], \
                      iris_df[iris_df['is_train'] == False]
        features = iris_df.columns[:4]  # extract 0 ~ 3 features
        # print(f'Iris features value: {features} \n')
        y = pd.factorize(train['species'])[0]
        # print(f'Iris y value: {y}')  # tot 3 species deduced
        # Iris y value: [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        #  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
        #  1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
        #  2 2 2 2 2 2 2]

        # Learning
        clf = RandomForestClassifier(n_jobs=2, random_state=0)
        clf.fit(train[features], y)
        # print(clf.predict_proba(test[features])[0:10])

        # Accuracy
        preds = iris.target_names[clf.predict(test[features])]
        # print(f'Iris crosstab Result: {preds[0:5]} \n')

        # Crosstab
        temp = pd.crosstab(test['species'], preds, rownames=['Actual Species'],
                           colnames=['Predicted Species'])
        # print(f'Iris crosstab Result: {temp} \n')
        # 0: setosa, 1: versicolor, 2: virginica

        # importance per feature / weights
        print(list(zip(train[features], clf.feature_importances_)))
        # [('sepal length (cm)', 0.08474010289429795),
        # ('sepal width (cm)', 0.022461263894393204),
        # ('petal length (cm)', 0.4464851467243143),
        # ('petal width (cm)', 0.4463134864869946)]

    def advanced(self):  # classifier
        iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                           header=None)
        # 0: setosa, 1: versicolor
        iris_mini = iris.iloc[0:100, 4]  # create 100 lines with 4 features
        y = np.where(iris_mini == 'Iris-setosa', -1, 1)  # binary classifying
        X = iris.iloc[0:100, [0, 2]].values  # proba var
        self.draw_scatter(X)
        # clf = Perceptron(eta=0.1, n_iter=10)  # 40yrs old code. IGNORE!
        # self.draw_decision_regions(X,y,classifier=clf, resolution=0.02)

    def draw_scatter(self, X):
        plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
        plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
        plt.xlabel('sepal length[cm]')
        plt.ylabel('petal length[cm]')
        plt.legend(loc='upper left')
        plt.savefig(f'{self.vo.context}iris_scatter.png')

    # def draw_decision_regions(self, X, y, classifier, resolution=0.02):
    #     # 마커와 컬러맵을 설정합니다
    #     markers = ('s', 'x', 'o', '^', 'v')
    #     colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    #     cmap = ListedColormap(colors[:len(np.unique(y))])
    #
    #     # 결정 경계를 그립니다
    #     x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    #     x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    #     """
    #     numpy 모듈의 arrange 함수는 반열린구간 [start, stop) 에서
    #     step 의 크기만큼 일정하게 떨어져 있는 숫자들을
    #     array 형태로 반환하는 함수
    #     meshgrid 함수는 사각형 영역을 구성하는
    #     가로축의 점들과 세로축의 점을
    #     나타내는 두 벡터를 인수로 받아서
    #     이 사각형 영역을 이루는 조합을 출력한다.
    #     결과는 그리드 포인트의 x 값만을 표시하는 행렬과
    #     y 값만을 표시하는 행렬 두 개로 분리하여 출력한다
    #     """
    #     xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    #                            np.arange(x2_min, x2_max, resolution))
    #     Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    #     Z = Z.reshape(xx1.shape)
    #     plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    #     plt.xlim(xx1.min(), xx1.max())
    #     plt.ylim(xx2.min(), xx2.max())
    #
    #     # 샘플의 산점도를 그립니다
    #     for idx, cl in enumerate(np.unique(y)):
    #         plt.scatter(x=X[y == cl, 0],
    #                     y=X[y == cl, 1],
    #                     alpha=0.8,
    #                     c=colors[idx],
    #                     marker=markers[idx],
    #                     label=cl,
    #                     edgecolor='black')
    #
    #     plt.xlabel('sepal length [cm]')
    #     plt.ylabel('petal length [cm]')
    #     plt.legend(loc='upper left')
    #     plt.savefig(f'{self.vo.context}iris_decision_region.png')
