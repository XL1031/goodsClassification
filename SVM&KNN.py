import re
import nltk
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import matplotlib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
'''
把字符串中的标点符号和数字全部转换为空格
输入：['70% of the wood in this unitco', 'kiln dried poplar lumber', 'basswood kd lumber  ddc is inc', 'newpage flash dried hardwood k', 'wooden chopsticks', 'canvas(wooden frame) contrca t no???', 'wooden box', "plywood 1220mmx2440mmx18mm freight prepaid'", '(ippc 1, plywood 5) parts for', 'usa round logs(douglas fir) 15', '24 logs european beech logs 12', 'spruce logs 93,46cbm 1166pc 14', 'new zealand radiata pine nett']
输出：['    of the wood in this unitco', 'kiln dried poplar lumber', 'basswood kd lumber  ddc is inc', 'newpage flash dried hardwood k', 'wooden chopsticks', 'canvas wooden frame  contrca t no   ', 'wooden box', 'plywood     mmx    mmx  mm freight prepaid ', ' ippc    plywood    parts for', 'usa round logs douglas fir    ', '   logs european beech logs   ', 'spruce logs      cbm     pc   ', 'new zealand radiata pine nett']
'''
def remove_symbols(listOfEveryLine):
    cutPunctuation = []
    del_estr = string.punctuation + string.digits  # ASCII 标点符号，数字
    # maketrans函数要求del_estr和replace长度相同，同一位置的字符相互匹配
    replace = " " * len(del_estr)
    # tran_tab是映射表，建立标点符号以及数字到空格的映射
    tran_tab = str.maketrans(del_estr, replace)
    for eachLine in listOfEveryLine:
        cutPunctuation.append(eachLine.translate(tran_tab))
    return cutPunctuation
'''
filename:需要读取的文件名，里面的数据每行都是字符串
test.txt文件内容：
-------------------------------
70% of the wood in this unitco
wooden chopsticks
kiln dried poplar lumber
-------------------------------
isReadLine为True：按行读取字符串，把每一行数据存入列表之中，例如读取文件后返回['70% of the wood in this unitco\n', 'wooden chopsticks\n', 'kiln dried poplar lumber\n']
isReadLine为False：整个读取文件，所有内容组成一个字符串，例如读取之后返回：'70% of the wood in this unitco\nwooden chopsticks\nkiln dried poplar lumber'
'''
def readFile(filename,isReadLine):
    openFile = open(filename,'r',encoding='utf-8')
    content = []
    #按照行获取文件内容，返回的是整个文件内容，每一行放入到一个列表之中
    if(isReadLine):
        for line in openFile:
            content.append(line)
    #否则所有内容放在一个列表之中
    else:
        content = openFile.read()
    openFile.close()
    return content
'''
把字符串按照空格分割
输入：[' of the wood in this unitco', 'wooden chopsticks', 'kiln dried poplar lumber']
输出：[['of', 'the', 'wood', 'in', 'this', 'unitco'], ['wooden', 'chopsticks'], ['kiln', 'dried', 'poplar', 'lumber']]
'''
def wordTokenize(ListOfContent):
    afterToken = []
    for everyStrLine in ListOfContent:
        # 加入set是为了去除重复，例如一行数据中wood可能出现多次，这里认为只出现一次#取消了去除重复
        afterToken.append(list(nltk.word_tokenize(everyStrLine)))
    return afterToken
'''
去除停用词，同时统计词频率///////取消了统计词频
输入：[['of', 'the', 'wood', 'in', 'this', 'unitco'], ['wooden', 'chopsticks'], ['kiln', 'dried', 'poplar', 'lumber']]
输出:
[['wood', 'unitco'], ['wooden', 'chopsticks'], ['kiln', 'dried', 'poplar', 'lumber']]
{'wood': 1, 'unitco': 1, 'wooden': 1, 'chopsticks': 1, 'kiln': 1, 'dried': 1, 'poplar': 1, 'lumber': 1}
'''
def cutStopWords(listOfStr,stopFilePath):
    readAllData = readFile(stopFilePath,True)
    readStopWords = []
    returnWords = []
    # 获取停用词列表
    for line in readAllData:
        readStopWords.append(line.strip())
    for line in listOfStr:
        wordsInLine = []
        for word in line:
            if word.lower() not in readStopWords:
                wordsInLine.append(word)
        returnWords.append(wordsInLine)
    return returnWords
def dataClean(contentOfEveryLine):
     # 去除标点和数字
     cutPunctuationed = remove_symbols(contentOfEveryLine)
     # 按照空格进行分词，分词之后去除重复单词，因为有些货名重复出现，例如'WODEN OAK,WODEN OAK,WODEN OAK'，这种情况会统计WODEN出错
     # 换句话说，就是只统计每一行货名中不重复的部分
     afterToken = wordTokenize(cutPunctuationed)
     # 去除停用词，统计词频
     afterCutStopWords= cutStopWords(afterToken,'stopWords.txt')
     # 升序排序
     return afterCutStopWords
def dataSavedToFile(outFileName,data):
    # 暂时存放元组类型
    with open(outFileName,'w',encoding='utf-8') as f:
        for wordFrequencyTuple in data:
            f.write(" ".join(str(t) for t in wordFrequencyTuple) + '\n') #换行写入，一行写入一个键值对
    f.close()
'''
创造数据集，将经过数据清理后的数据集调整格式，返回值为DataFrame格式的特征data和标签label
'''
def constructDataset(write_path):
    df = pd.read_excel(write_path)
    goodsname = df['货名']
    corpus_df = dataClean(goodsname)#格式list套list   例如[['1','2','3'],['4','5','6']]
    label_df = df['二类代码']        #格式 dataFrame
    #corpus_df格式不符合CountVectorizer()，用列表将corpus_df中的数据连接起来，并存入L[],然后将L赋值给新的DataFrame变量data
    L = []
    for i in corpus_df:
        L.append(" ".join(i))
    data= pd.DataFrame(L,columns=['货名'])
    label = label_df
    label.columns = '二类代码'
    return data, label
def SVMModel(Data_path):
    df1, df2 = constructDataset(Data_path)
    # 特征提取
    countvec = CountVectorizer()
    data = countvec.fit_transform(df1["货名"])
    label = df2
    # 训练集切分
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=40)
    # 支持向量机
    SVM_model = SVC()
    SVM_model.fit(x_train, y_train)
    # 训练结果
    print(SVM_model.score(x_test, y_test))
def KNNModel(Data_path):
    k = 20  # K的取值范围
    df1, df2 = constructDataset(Data_path)
    # 特征提取
    countvec = CountVectorizer()
    data = countvec.fit_transform(df1["货名"])
    label = df2
    # 训练集切分
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=40)
    #KNN
    knn_distortions=[]
    for i in range(1, k, 1):
        knn_model = KNeighborsClassifier(n_neighbors=i)
        knn_model.fit(x_train, y_train)
        print(
            "-------------------------------------设置K为%d时的分类情况-------------------------------------------" % i)
        knn_accuracy = knn_model.score(x_test,y_test)
        print('分类准确率为：', knn_accuracy)
        knn_distortions.append(knn_accuracy)
    #画图表示分类准确率随K的增加而变化
    X, Y = [], []
    for i in range(1, k):
         X.append(i)
         Y = knn_distortions
    plt.plot(X, Y, '-p', color='grey',
             marker='o',
             markersize=8, linewidth=2,
             markerfacecolor='red',
             markeredgecolor='grey',
             markeredgewidth=2)
    plt.show()

if __name__ == "__main__":

     #SVMModel('goodsData.xlsx')
     KNNModel('goodsData.xlsx')
