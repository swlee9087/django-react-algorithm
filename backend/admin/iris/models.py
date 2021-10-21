from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from admin.common.models import ValueObject
from admin.tensor.models import Perceptron


class Iris(object):
    def __init__(self):
        self.vo = ValueObject()
        self.vo.context = 'admin/iris/data/'

    # def hook(self):  # too much then write hook
    #     self.base()

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
        X = iris.iloc[0:100, [0,2]].values  # proba var
        clf = Perceptron(eta=0.1, n_iter=10)
        self.draw_scatter(X)

    def draw_scatter(self, X):
        plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
        plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
        plt.xlabel('sepal length[cm]')
        plt.ylabel('petal length[cm]')
        plt.legend(loc='upper left')
        plt.savefig(f'{self.vo.context}iris_scatter.png')


