import re
import nltk
import pandas as pd
import string
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
        # 加入set是为了去除重复，例如一行数据中wood可能出现多次，这里认为只出现一次
        afterToken.append(list(set(nltk.word_tokenize(everyStrLine))))
    return afterToken
'''
去除停用词，同时统计词频率
输入：[['of', 'the', 'wood', 'in', 'this', 'unitco'], ['wooden', 'chopsticks'], ['kiln', 'dried', 'poplar', 'lumber']]
输出:
[['wood', 'unitco'], ['wooden', 'chopsticks'], ['kiln', 'dried', 'poplar', 'lumber']]
{'wood': 1, 'unitco': 1, 'wooden': 1, 'chopsticks': 1, 'kiln': 1, 'dried': 1, 'poplar': 1, 'lumber': 1}
'''
def cutStopWords(listOfStr,stopFilePath):
    wordFrequency = {}
    readAllData = readFile(stopFilePath,True)
    readStopWords = []
    returnWords = []
    # 获取停用词列表
    for line in readAllData:
        readStopWords.append(line.strip())
    for line in listOfStr:
        wordsInLine = []
        for word in line:
            if word not in readStopWords:
                wordFrequency[word] = wordFrequency.get(word,0) + 1
                wordsInLine.append(word)
        returnWords.append(wordsInLine)
    return returnWords,wordFrequency
'''
该函数用于从初始Excel文件中提取某个二级分类对应的所有货名，每一行的货物名称装入一个列表
dataFrameFromHigh：指使用pandas从Excel中读取返回的DataFrame
columnName：表格某一列的列名，例如'二级代码','二级名称','一级代码'，'一级名称'
targetColumn：需要获取的某一列数据的列名，例如需要获得货名，则传入'货名',需要获得'一级名称'，则传入'一级名称'
twoClassCode：指的是某一个二级分类的分类号，例如木材的二级分类包括44、4407、4412等
'''
def getAllGoodsNameOfOneClassCode(dataFrameFromHigh,columnName,targetColumn,twoClassCode):
    dataFrame = dataFrameFromHigh
    # targetDataFrame数据类型也是DataFrame，即表格中分类代码与twoClassCode相同的行组成新的表格
    targetDataFrame = dataFrame.loc[dataFrame[columnName]==twoClassCode]
    targetColumnIndex = targetDataFrame.columns.get_loc(targetColumn)
    targetNameList = []
    for line in targetDataFrame.values:
        if type(line[targetColumnIndex]) == str:
            targetNameList.append(line[targetColumnIndex].lower())
        else:
            targetNameList.append(line[targetColumnIndex])
    return targetNameList
'''
获取所有二级分类代码对应的货物名称，返回一个Map
:数据集的路径
columnName：表格某一列的列名，例如'二级代码','二级名称','一级代码'，'一级名称'
targetColumn：需要获取的某一列数据的列名，例如需要获得货名，则传入'货名',需要获得'一级名称'，则传入'一级名称'
twoClassCode：指的是某一个二级分类的分类号，例如木材的二级分类包括44、4407、4412等
返回:Map[key] = [all lines of the goodsname of the classCode ]，key是二级分类的取值，例如44、7704、7712等
'''
def getAllGoodsNameOfTheColumnWithClassCode(excelPath,columnName,targetColumn,classCode):
    # 首先获取表格数据的DataFrame
    dataFrame = pd.read_excel(excelPath)
    # 获取columnName下，分类代码为classCode的某一个targetColumnContent的内容
    # 从dataFrame表格句柄中获取所有分类代码列表，unique()函数用于去除重复的代码
    TwoGradeCodeList = dataFrame[columnName].unique()
    # Map：{code: []}，即分级代码所对应的所有货物名称，每行货物名称被放入一个列表，该列表存入一个大的列表之中。
    # Map样例：{4407: [['70% of the wood in this unitco'], ['kiln dried poplar lumber']]}
    codeMapGoodsName = {}
    if classCode=="All":
        # 字典键All存储该分类下所有二级代码对应的货物名，所有货物名被装进一个列表之中
        codeMapGoodsName['All'] = []
        for code in TwoGradeCodeList:
            returnList = getAllGoodsNameOfOneClassCode(dataFrame,columnName,targetColumn, code)
            codeMapGoodsName[code] = returnList
            codeMapGoodsName.get('All').extend(returnList)
    else:
        codeMapGoodsName[classCode] = getAllGoodsNameOfOneClassCode(dataFrame,columnName,targetColumn,classCode)
    return codeMapGoodsName
'''
清除所有标点符号、按照空格分词、统计单词频率
contentOfEveryLine：某一二级分类所对应的所有货名，[1,2,3,4]，每个数字代表一行货名的字符串
例子：
输入：['wooden chopsticks', 'canvas(wooden frame) contrca t no???', 'wooden box']
输出：[('wooden', 3), ('chopsticks', 1), ('canvas', 1), ('frame', 1), ('contrca', 1), ('box', 1)]
'''
def dataCleanAndStatisticsWordFrequency(contentOfEveryLine):
     # 去除标点和数字
     cutPunctuationed = remove_symbols(contentOfEveryLine)
     # 按照空格进行分词，分词之后去除重复单词，因为有些货名重复出现，例如'WODEN OAK,WODEN OAK,WODEN OAK'，这种情况会统计WODEN出错
     # 换句话说，就是只统计每一行货名中不重复的部分
     afterToken = wordTokenize(cutPunctuationed)
     # 去除停用词，统计词频
     afterCutStopWords,wordsFrequency = cutStopWords(afterToken,'data/stopWords.txt')
     # 升序排序
     return sorted(wordsFrequency.items(),key=lambda x:x[1],reverse=True)
def dataSavedToFile(outFileName,data):
    # 暂时存放元组类型
    with open(outFileName,'w',encoding='utf-8') as f:
        for wordFrequencyTuple in data:
            f.write(" ".join(str(t) for t in wordFrequencyTuple) + '\n') #换行写入，一行写入一个键值对
    f.close()
'''
数据清理、把排序后词频存入文件中
excelPath：excel文件路径
columnName：列名
targetColumn：要获取的目标列
classCode：对应列名的分类代码，如果不指明该参数，则默认使用All，即获取所有columnName的分类代码当作classCode
'''
def dataOfOneOrAllClassCodeSavedToFile(excelPath,columnName,targetColumn,classCode='All'):
    codeMapTargetData = getAllGoodsNameOfTheColumnWithClassCode(excelPath,columnName,targetColumn,classCode)
    if classCode=="All":
        outFileName = columnName +"_所有"+columnName+"_"+targetColumn+"词频.txt"
        returnData = dataCleanAndStatisticsWordFrequency(codeMapTargetData.get('All'))
    else:
        outFileName = columnName+"_"+str(classCode)+"_"+targetColumn+"词频.txt"
        returnData = dataCleanAndStatisticsWordFrequency(codeMapTargetData.get(classCode))
    # 接下来输出至文件，文件名为：columnName_classCode_targetColumn
    dataSavedToFile(outFileName,returnData)
if __name__ == "__main__":
    df = pd.read_excel('data/goodsData.xlsx')
    twoClassCodeList = df['二类代码'].unique()
    for code in twoClassCodeList:
         dataOfOneOrAllClassCodeSavedToFile('data/goodsData.xlsx','二类代码','货名',code)
    dataOfOneOrAllClassCodeSavedToFile('data/goodsData.xlsx','二类代码','货名','All')

