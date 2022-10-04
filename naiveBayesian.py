from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
# from dataCleanFunction import *
# import random
from tf_itfModel import *
from sklearn.metrics import recall_score
def dataTranslate(dataMap):
    '''
   把字典数据转化为列表
    :param dataMap: 例如：{4407: ['70% of the wood in this unitco', 'kiln dried poplar lumber'],4403:['red oak lumber, kd (12621 bf)','newpage flash dried hardwood k'}
    :return: dataList: ['70% of the wood in this unitco', 'kiln dried poplar lumber', 'red oak lumber, kd (12621 bf)', 'newpage flash dried hardwood k']
             dataLabel:[4407,4407,4403,4403]
    '''
    dataList = []
    dataLabel = []
    for key in dataMap:
        dataList.extend(dataMap.get(key))
        for i in range(len(dataMap.get(key))):
            dataLabel.append(key)
    return dataList,dataLabel
def dataSetSplit(data):
    '''
    数据集划分，把每一类数据的80%用于训练，20%用于测试
    :param data:
    :return:
    '''
    trainMap = {}
    testMap = {}
    for key in data.keys():
        if key != 'All':
            # 测试集占比20%
            testMap[key], trainMap[key] = data_split(data.get(key), 0.2, True)
            # 划分完训练集和测试集之后，把训练集每一类数据组合起来，传入向量空间
    trainList,trainLabel = dataTranslate(trainMap)
    testList,testLabel = dataTranslate(testMap)
    return trainList,trainLabel,testList,testLabel
def dataClean(dataList):
    '''
    数据清理和转化，转化成贝叶斯模型输入
    :param dataList:['70% of the wood in this unitco', 'kiln dried poplar lumber', 'red oak lumber, kd (12621 bf)', 'newpage flash dried hardwood k']
    :return:['wood unitco', 'dried lumber poplar kiln', 'lumber oak bf red kd', 'flash newpage dried hardwood']
    '''
    dataCutStopWords, wordFrequency = dataCleanAndStatisticsWordFrequency(dataList)
    dataContent = list2str(dataCutStopWords)
    return dataContent
def dataBalence(data,num):
    '''
    数据平衡策略，把小于500条的数据重复到500条
    :param data:
    :return:添加重复数据之后的data
    '''
    balencedData = {}
    for key in data.keys():
        if key != 'All':
            length = len(data.get(key))
            if length < num:
                freq = int(num/length)
                contentList = data.get(key)
                newContent = []
                for i in range(freq):
                    newContent.extend(contentList)
                balencedData[key] = newContent
    return balencedData
if __name__ == "__main__":
    # 读取数据
    data = getAllGoodsNameOfTheColumnWithClassCode('data/test.xlsx', '二类代码', '货名', 'All')
    # 数据平衡，把样本数量小于500条的数据进行重复，把数据重复500/样本数量次，把数据量升高到500
    data = dataBalence(data,500)
    # 划分训练集和测试集
    trainList,trainLabel,testList,testLabel = dataSetSplit(data)
    # 数据清理
    trainAfterClean = dataClean(trainList)
    testAfterClean = dataClean(testList)
    # 词频模型
    count_vector = CountVectorizer()
    trainVectorMatrix = count_vector.fit_transform(trainAfterClean)
    # print(count_vector.get_feature_names_out())  # 看到所有文本的关键字
    # print(count_vector.vocabulary_)  # 文本的关键字和其位置
    # print(trainVectorMatrix.toarray())  # 词频矩阵的结果
    # 获取训练集合tf-itf矩阵
    train_tfidf = TfidfTransformer(use_idf=False).fit_transform(trainVectorMatrix)
    # print(train_tfidf)
    # 训练贝叶斯模型，多项式和伯努利
    # MUB = MultinomialNB().fit(train_tfidf, trainLabel)
    MUB = BernoulliNB().fit(train_tfidf, trainLabel)
    # 测试贝叶斯模型
    testVector = count_vector.transform(testList)  # 得到测试集的词频矩阵
    # 用transformer.fit_transform(词频矩阵)得到TF权重矩阵
    test_tfidf = TfidfTransformer(use_idf=False).fit_transform(testVector)
    predict = MUB.predict(test_tfidf)
    print(MUB.score(test_tfidf, testLabel))
    print(recall_score(testLabel, predict, average='macro'))

