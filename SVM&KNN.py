from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC, NuSVC
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from dataCleanFunction import *
'''
创造数据集，将经过数据清理后的数据集调整格式，返回值为DataFrame格式的特征df1和标签df2
*********这样构造数据集不正确，划分的是整个数据集，不是每个分类的数据集，会造成数据不平衡、预测不准确
'''
def constructDataset(write_path):
    df = pd.read_excel(write_path)
    goodsname = df['货名']
    corpus_df,wordFrequency= dataCleanAndStatisticsWordFrequency(goodsname)#格式list套list   例如[['1','2','3'],['4','5','6']]
    label_df = df['二类代码']        #格式 dataFrame
    #corpus_df格式不符合CountVectorizer()，用列表将corpus_df中的数据连接起来，并存入L[],然后将L赋值给新的DataFrame变量data
    L = []
    for i in corpus_df:
        L.append(" ".join(i))
    data= pd.DataFrame(L,columns=['货名'])
    label = label_df.to_frame(name='二类代码')
    return data.values.ravel(), label.values.ravel() #将DataFrame格式转换为一维数组
def SVMModel(Data_path):
    df1, df2 = constructDataset(Data_path)
    # 特征提取
    countvec = CountVectorizer()
    data = countvec.fit_transform(df1)
    label = df2
    # 训练集切分
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=40)
    # 支持向量机
    SVM_model = LinearSVC()
    SVM_model.fit(x_train, y_train)
    # 训练结果
    print(SVM_model.score(x_test, y_test))
    #LinearSVC 0.9562150055991041
    #SVC       0.9310190369540874
def KNNModel(Data_path):
    k = 20  # K的取值范围
    df1, df2 = constructDataset(Data_path)
    # 特征提取
    countvec = CountVectorizer()
    data = countvec.fit_transform(df1)
    label = df2
    # 训练集切分
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=40)
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
    #SVMModel('data/test.xlsx')
    KNNModel('data/test.xlsx')
