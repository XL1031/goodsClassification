'''
这里是tf-itf模型来预测分类，主要思路如下：
1、划分数据集，训练集80%，测试集20%。
2、把训练集组合成长文本建立字典，然后使用该字典建立tf-itf模型，并构建每一个二级分类的向量空间，每个二级分类矩阵的维度是(数据行数，字典空间大小)
无法和一行数据进行相似性比较，所以需要把二级分类降低维度，暂时采用矩阵取平均值的方式，返回一维矩阵，维度为(1，字典空间大小)。
3、把测试数据在训练数据的背景下转化为向量空间，即假设我们已经知道测试数据是木材类，直接在木材类数据的tf-itf模型下生成向量，把每一行的测试数据
分别和训练数据的平均矩阵使用余弦相似度计算，结果越高则准确率越高
关键参数：text2vec/text2vec中的_get_docs_dict函数中的过滤函数，保留至少出现no_below次的单词，删除超过在no_above个文档中出现的单词
bobelow:1,no_above:1 precision: 63%
'''
import numpy as np
from text2vec import text2vec
from dataCleanFunction import *
import random
def data_split(full_list, ratio, shuffle=False):
    '''
    数据集拆分函数: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio: 比率，即划分测试集的占比
    :param shuffle: 是否使用洗牌函数随机抽取数据
    :return:
    '''
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2

def list2str(datalist):
    '''
    把列表数据转换为字符串数据
    :param datalist: 数据列表，例如 [['core', 'film', 'combi', 'faced', 'plywood'], ['maritime', 'pine']]
    :return: ['core film combi faced plywood', 'maritime pine', 'ayous lumber edged square']
    '''
    strContent = []
    for line in datalist:
        strContent.append(' '.join(line))
    return strContent

def data2Vector(is_train,dataMap,d2v):
    '''
    tf-itf核心函数，把数据转化为向量表示
    :param is_train:判断是否是训练集，训练集和测试集返回值不一样，训练集先转化为向量集合，再把一类的向量集合平均化为一维向量
                    而测试集只是转化为向量，使用转化完的向量与训练集一维向量相乘得到相似度
    :param dataMap:数据集合，例如[['core', 'film', 'combi', 'faced', 'plywood'], ['maritime', 'pine']]
    :param d2v: 向量类，其中包含建立tf-itf模型的函数
    :return: 处理完成的向量表示
    '''
    # 如果是训练集合，获取向量表示并把一类数据转化为一个行向量表示
    key2Vec = {}
    if is_train:
        # 对每一个二级分类，把数据清理之后传入tf-itf模型，获得其向量表示，并平均为一维向量
        for key in dataMap.keys():
            if key != 'All':
                cleanedData, wordFrequency = dataCleanAndStatisticsWordFrequency(dataMap.get(key))
                strData = list2str(cleanedData)
                key2Vec[key] = (np.mean(d2v.get_tfidf(strData),axis=0)).reshape(1, len(d2v.docs_dict))
    else:
        for key in dataMap.keys():
            # 有些类可能没有数据
            if key != 'All' and len(dataMap.get(key)) != 0:
                cleanedTestData, testWodrFrequency = dataCleanAndStatisticsWordFrequency(dataMap.get(key))
                testStrData = list2str(cleanedTestData)
                key2Vec[key] = d2v.get_tfidf(testStrData)
    return key2Vec
def predictTest(test2VecMap,train2VecMeanMap):
    # 预测正确的数据行
    predictTrue = 0
    # 数据总行数
    testDataSize = 0
    # 获取二级分类的所有向量数据
    for testKey in test2VecMap:
        # 获取二级分类的某一行向量数据
        for vec in test2VecMap.get(testKey):
            testDataSize = testDataSize + 1
            reshapeTestVec = vec.reshape(1, len(d2v.docs_dict))
            reshapeTestVecMap2predictKey = {}
            for predictKey in train2VecMeanMap:
                reshapeTestVecMap2predictKey[predictKey] = text2vec.simical(reshapeTestVec[0],train2VecMeanMap.get(predictKey)[0]).Cosine()
            maxPredictKey = max(reshapeTestVecMap2predictKey.items(), key=lambda x: x[1])[0]
            if maxPredictKey == testKey:
                predictTrue = predictTrue + 1
    print("PredictTrue:" + str(predictTrue))
    print("testDataSize:" + str(testDataSize))
    print("precision:" + str(predictTrue / testDataSize))

if __name__ == "__main__":
    # 读取数据
    data = getAllGoodsNameOfTheColumnWithClassCode('data/test.xlsx', '二类代码', '货名', 'All')
    # 划分训练集和测试集
    train_data_Map = {}
    test_data_Map = {}
    for key in data.keys():
        if key != 'All':
            # 测试集占比20%
            test_data_Map[key], train_data_Map[key] = data_split(data.get(key), 0.2, True)
    # 划分完训练集和测试集之后，把训练集每一类数据组合起来，传入向量空间
    allDataContent = []
    for key in train_data_Map:
        allDataContent.extend(train_data_Map.get(key))
    # allData数据清理
    allDataCutStopWords, wordFrequency = dataCleanAndStatisticsWordFrequency(allDataContent)
    allContent = list2str(allDataCutStopWords)
    # 接下来建立向量空间字典
    d2v = text2vec.text2vec(allContent)
    # 获取该向量空间的tf-itf模型
    allDataVector = d2v.get_tfidf()
    test2VecMap = data2Vector(False, test_data_Map, d2v)
    train2VecMeanMap = data2Vector(True, train_data_Map, d2v)
    predictTest(test2VecMap, train2VecMeanMap)

