from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC, NuSVC
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from dataCleanFunction import *
from tf_itfModel import data_split, list2str

'''
创造数据集，将经过数据清理后的数据集调整格式，返回值为DataFrame格式的特征df1和标签df2
*********这样构造数据集不正确，划分的是整个数据集，不是每个分类的数据集，会造成数据不平衡、预测不准确
修改后的代码按照‘二类代码’划分数据集和测试集
'''
def constructDataset(write_path,columnName,targetColumn,classCode):
    # 读取数据
    data = getAllGoodsNameOfTheColumnWithClassCode('data/test.xlsx', '二类代码', '货名', 'All')
    # 划分训练集和测试集
    train_data_Map = {}
    test_data_Map = {}
    for key in data.keys():
        if key != 'All':
            # 测试集占比20%
            test_data_Map[key], train_data_Map[key] = data_split(data.get(key), 0.2, True)
    # 划分完训练集和测试集之后，把训练集每一类数据组合起来

    trainDataList = []
    trainLabelList = []
    testDataList  = []
    testLabelList = []
    for key,value in train_data_Map.items():
        for line in value:
            trainLabelList.append(key)
            trainDataList.append(line)
    for key,value in test_data_Map.items():
        for line in value:
            testLabelList.append(key)
            testDataList.append(line)
    # 数据清理
    trainDataCutStopWords, wordFrequency = dataCleanAndStatisticsWordFrequency(trainDataList)
    testDataCutStopWords, wordFrequency  = dataCleanAndStatisticsWordFrequency(testDataList)

    print(testLabelList)
    return trainDataList,trainLabelList,testDataList,testLabelList


def SVMModel(Data_path,columnName,targetColumn,classCode):
    l1, l2, l3, l4 = constructDataset(Data_path,columnName,targetColumn,classCode)
    # 特征提取
    countvec = CountVectorizer()
    x_train = countvec.fit_transform(l1)
    y_train = l2
    x_test = countvec.transform(l3)
    y_test = l4
    SVM_model = LinearSVC(max_iter=200000)
    SVM_model.fit(x_train, y_train)
    # 训练结果
    print(SVM_model.score(x_test, y_test))
    #LinearSVC 0.9526050420168067
    #SVC       0.9225770308123249
def KNNModel(Data_path,columnName,targetColumn,classCode):
    k = 20  # K的取值范围
    l1, l2, l3, l4 = constructDataset(Data_path,columnName,targetColumn,classCode)
    # 特征提取
    countvec = CountVectorizer()
    x_train = countvec.fit_transform(l1)
    y_train = l2
    x_test = countvec.transform(l3)
    y_test = l4
    #KNN
    knn_distortions=[]
    for i in range(1, k, 1):
        knn_model = KNeighborsClassifier(n_neighbors=i)
        knn_model.fit(x_train, y_train)
        print( "-------------------------------------设置K为%d时的分类情况-------------------------------------------" % i)
        knn_accuracy = knn_model.score(x_test,y_test)
        print('分类准确率为：', knn_accuracy)
        knn_distortions.append(knn_accuracy)
    #画图表示分类准确率随K的增加而变化
    X, Y = [], []
    for i in range(1, k):
        X.append(i)
        Y = knn_distortions
    plt.plot(X, Y, '-p', color='grey', marker='o', markersize=8, linewidth=2, markerfacecolor='red', markeredgecolor='grey', markeredgewidth=2)
    plt.show()
if __name__ == "__main__":
    SVMModel('data/test.xlsx','货名','二类代码','All')
    #KNNModel('data/test.xlsx','货名','二类代码','All')
