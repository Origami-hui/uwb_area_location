import os
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import glob
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from sklearn.model_selection import train_test_split


def train_model():
    # 1. 创建分类器
    clf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    # 2. 训练模型
    clf.fit(X_train, y_train)

    execute_test(clf)
    # 保存模型
    # 假设 clf 是已训练好的随机森林模型
    joblib.dump(clf, 'random_forest_model_test.starry')


def test_model():
    # 加载模型
    clf_loaded = joblib.load('random_forest_model_test.starry')

    # 进行预测
    execute_test(clf_loaded)


def execute_test(clf):
    # 3. 使用测试集进行预测
    y_pred = clf.predict(X_test)

    # 4. 评估模型性能
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型准确率: {accuracy:.2f}")

    # 5. 进一步查看分类报告和混淆矩阵
    print("分类报告:")
    print(classification_report(y_test, y_pred))

    print("混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == '__main__':

    file_paths = glob.glob(os.path.join('nlos dataset', '*.csv'))

    # 读取并合并所有 CSV 文件
    df_list = [pd.read_csv(file) for file in file_paths]  # 逐个读取
    df = pd.concat(df_list, ignore_index=True)  # 合并所有 DataFrame

    # 查看数据
    print(df.head())

    # 分离特征和目标变量（假设目标变量名为 'target'）
    y = df[df.columns[4]]
    features = df.columns[1:3]
    X = df[features]  # 目标变量

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    # print(X_train, y_train)

    train_model()
    # test_model()
