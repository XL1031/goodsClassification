from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from dataCleanFunction import *

'''
创造数据集，将经过数据清理后的数据集调整格式，返回值为DataFrame格式的特征df1和标签df2
*********这样构造数据集不正确，划分的是整个数据集，不是每个分类的数据集，会造成数据不平衡、预测不准确
'''
def constructDataset(write_path):

    df = pd.read_excel('data/test.xlsx')
    goodsname = df['货名']
    # 格式list套list   例如[['1','2','3'],['4','5','6']]
    corpus_list,wordFrequency = dataCleanAndStatisticsWordFrequency(goodsname)
    label_list = df['二类代码']
    # corpus_list格式不符合CountVectorizer()，用列表将corpus_list中的数据连接起来，并存入L[],然后将L赋值给新的DataFrame变量df1
    L = []
    for i in corpus_list:
        L.append(" ".join(i))
    df1= pd.DataFrame(L,columns=['货名'])
    df2 = label_list
    df2.columns = '二类代码'
    return df1, df2

if __name__ == "__main__":
    df1,df2= constructDataset('data/test.xlsx')
    # 特征提取
    countvec = CountVectorizer()
    countvec = countvec.fit_transform(df1["货名"])
    # 训练集切分
    x_train, x_test, y_train, y_test = train_test_split(countvec, df2, test_size=0.2,random_state=40)
    model = SVC()
    model.fit(x_train, y_train)
    # 训练结果
    print(model.score(x_test, y_test))

