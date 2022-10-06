'''
使用库文件挖掘出一些关联规则出来，但是很多关联规则是重复的，也有冲突的，不好用于文本分类
'''
from naiveBayesian import*
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder  # 传入模型的数据需要满足特定的格式，可以用这种方法来转换为bool值，也可以用函数转换为0、1
from sklearn.metrics import recall_score
def dataAddLabel(data,trainLabel):
    '''
    用于在数据中插入标签名称
    :param data: ['oak bf red kd lumber', 'ddc kd lumber basswood', 'wood unitco', 'treated sawn wood raugh paint planned size']
    :param trainLabel:[4407, 4407, 4407, 4407]
    :return [['oak', 'bf', 'red', 'kd', 'lumber', '4407'], ['ddc', 'kd', 'lumber', 'basswood', '4407'], ['wood', 'unitco', '4407'], ['treated', 'sawn', 'wood', 'raugh', 'paint', 'planned', 'size', '4407']]
    '''
    length = len(data)
    for i in range(length):
        data[i] = (data[i] + " " + str(trainLabel[i])).split()
    return data
if __name__ == "__main__":
    # 读取数据
    data = getAllGoodsNameOfTheColumnWithClassCode('data/test1.xlsx', '二类代码', '货名', 'All')
    # 数据平衡，把样本数量小于500条的数据进行重复，把数据重复500/样本数量次，把数据量升高到500
    data = dataBalence(data,10)
    # 划分训练集和测试集
    trainList, trainLabel, testList, testLabel = dataSetSplit(data)
    # 数据清理
    trainAfterClean = dataClean(trainList)
    testAfterClean = dataClean(testList)
    data = dataAddLabel(trainAfterClean,trainLabel)
    te = TransactionEncoder()  # 使用独热编码类型
    df_tf = te.fit_transform(data)
    # 将 True、False 转换为 0、1 # 官方给的其它方法
    # df_01 = df_tf.astype('int')
    # 将编码值再次转化为原来的商品名
    # df_name = te.inverse_transform(df_tf)
    # te.columns_是data中的所有单词的列表
    df = pd.DataFrame(df_tf, columns=te.columns_)
    print(df)

    # use_colnames=True表示使用元素名字，默认的False使用列名代表元素
    frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
    # frequent_itemsets = apriori(df,min_support=0.05)
    # 频繁项集按支持度排序
    # frequent_itemsets.sort_values(by='support', ascending=False, inplace=True)
    # 选择长度 >=2 的频繁项集
    # print(frequent_itemsets[frequent_itemsets.itemsets.apply(lambda x: len(x)) >= 2])
    # metric可以有很多的度量选项，返回的表列名都可以作为参数
    association_rule = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.9)
    # 关联规则可以按leverage排序
    association_rule.sort_values(by='leverage', ascending=False, inplace=True)
    association_rule.to_csv("associationRules.csv")