from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from naiveBayesian import*
from sklearn.metrics import recall_score
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
    # 训练决策树
    DecisionTree = DecisionTreeClassifier(criterion="gini", splitter="random", max_depth=None, min_samples_split=2, min_samples_leaf=2)
    DecisionTree.fit(train_tfidf, trainLabel)
    # 测试决策树模型
    testVector = count_vector.transform(testList)  # 得到测试集的词频矩阵
    # 用transformer.fit_transform(词频矩阵)得到TF权重矩阵
    test_tfidf = TfidfTransformer(use_idf=False).fit_transform(testVector)
    predict = DecisionTree.predict(test_tfidf)
    print(DecisionTree.score(test_tfidf, testLabel))
    print(recall_score(testLabel, predict,average='macro'))