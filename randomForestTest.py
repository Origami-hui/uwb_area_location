from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import joblib


def train_model():
    clf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    y, _ = pd.factorize(train['species'])
    clf.fit(train[features], y)

    preds = iris.target_names[clf.predict(test[features])]
    # 计算预测准确性
    accuracy = accuracy_score(test['species'], preds)
    print(f'Accuracy: {accuracy:.2f}')

    # 创建混淆矩阵
    confusion_matrix = pd.crosstab(test['species'], preds, rownames=['actual'], colnames=['preds'])
    print(confusion_matrix)

    # 保存模型
    # 假设 clf 是已训练好的随机森林模型
    joblib.dump(clf, 'random_forest_model_test.starry')


def test_model():
    # 加载模型
    clf_loaded = joblib.load('random_forest_model_test.starry')

    # 进行预测
    preds = iris.target_names[clf_loaded.predict(test[features])]
    # 计算预测准确性
    accuracy = accuracy_score(test['species'], preds)
    print(f'Accuracy: {accuracy:.2f}')


if __name__ == '__main__':
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    print(df.head())

    # 训练集、测试集分类
    train, test = df[df['is_train'] == True], df[df['is_train'] == False]
    features = df.columns[:4]

    #train_model()
    test_model()
