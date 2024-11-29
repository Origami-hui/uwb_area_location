import os
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import glob
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from config import *
import numpy as np


def train_model():
    # 1. 创建分类器
    clf = RandomForestClassifier(n_estimators=100, n_jobs=1)
    # 2. 训练模型
    clf.fit(X_train, y_train)

    execute_test(clf, X_val, y_val)
    # 保存模型
    # 假设 clf 是已训练好的随机森林模型
    joblib.dump(clf, NLOS_MODEL_NAME)


def test_model(_X, _y):
    # 加载模型
    clf_loaded = joblib.load(NLOS_MODEL_NAME)

    # 进行预测
    print("X_test", X_test)
    execute_test(clf_loaded, _X, _y)


def execute_test(clf, X, y):
    # 3. 使用测试集进行预测
    y_pred = clf.predict(X)

    # 4. 评估模型性能
    accuracy = accuracy_score(y, y_pred)
    print(f"模型准确率: {accuracy:.2f}")

    # 5. 进一步查看分类报告和混淆矩阵
    print("分类报告:")
    print(classification_report(y, y_pred))

    print("混淆矩阵:")
    print(confusion_matrix(y, y_pred))

    # fpr, tpr, thresholds = roc_curve(y, y_pred)
    # print(fpr, tpr)


def random_test():
    num_classes = len(np.unique(y_test))  # 获取类别数量
    y_pred_random = np.random.randint(0, num_classes, size=y_test.shape)  # 生成随机预测
    accuracy_random = accuracy_score(y_test, y_pred_random)
    print(f"随机预测准确率: {accuracy_random:.2f}")

    print("随机预测分类报告:")
    print(classification_report(y_test, y_pred_random))

    print("随机预测混淆矩阵:")
    print(confusion_matrix(y_test, y_pred_random))


if __name__ == '__main__':

    file_paths = glob.glob(os.path.join('nlos dataset', '*.csv'))
    # file_paths = glob.glob(os.path.join('nlos dataset', '*.csv'))
    print(file_paths)
    # 读取并合并所有 CSV 文件
    df_list = [pd.read_csv(file) for file in file_paths]  # 逐个读取
    df = pd.concat(df_list, ignore_index=True)  # 合并所有 DataFrame

    # 查看数据
    print(df.head())

    # 分离特征和目标变量（假设目标变量名为 'target'）
    y = df[df.columns[4]]
    features = df.columns[1:4]
    X = df[features]  # 目标变量

    # counts = y.value_counts()
    #
    # # 获取 0 和 1 的个数
    # count_0 = counts.get(0, 0)  # 防止没有 0 时抛出 KeyError
    # count_1 = counts.get(1, 0)  # 防止没有 1 时抛出 KeyError
    #
    # print(f"0 的个数: {count_0}")
    # print(f"1 的个数: {count_1}")

    test_flag = False

    if test_flag:

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        test_model(X_test, y_test)
        # random_test()

    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
        train_model()
