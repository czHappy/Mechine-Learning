import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
from collections import Counter

def set_show():
    #显示所有列 所有行
    pd.set_option('display.width',1000)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

def read_data(file='data.txt', sep=' '):
    # 导入数据
    data = pd.read_csv(file, sep=',')
    GT = np.array(data.values[:, -1]) #取最后一列
    # print(GT)
    columns = data.columns.values.tolist()[0:-1] # 取列名列表
    data = data[columns] # 取数据矩阵
    return data, columns, GT


def train(data, eps=0.5, min_samples=5):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)  # 设置半径为10eps，最小样本量为min_samples，建模
    preds = db.labels_
    return preds

def results_analysis(data, clas, GT, preds):
    data['cluster_gt'] = GT
    data['cluster_db'] = preds # -1 0 1 2 3....
    data.sort_values('cluster_db')

    print(data)
    print(data.groupby('cluster_db').mean())

    # 统计噪声率
    noise_ratio = 1.0 * Counter(preds)[-1] / len(preds)
    print('noise_ratio = ', noise_ratio)

    # 统计准确率\
    T = 0
    for i in range(len(GT)):
        if clas.index(GT[i]) == preds[i]: #这里有一点冒险 因为分的cluster序号不一定是按照class列表的顺序
            T = T + 1
    accuracy = T / len(preds)
    print('accuracy = ', accuracy)

set_show()
file = 'iris.data'
data, columns, GT = read_data(file, sep=' ')

# print(columns)
# print(data)
# print(GT)

preds = train(data, 0.4242, 5) #0.4242 5
print(preds)
clas = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
results_analysis(data, clas, GT, preds)


# # 画出在不同两个指标下样本的分布情况
# #print(pd.plotting.scatter_matrix(X, c=beer.cluster_db.tolist(), figsize=(10,10), s=100))
