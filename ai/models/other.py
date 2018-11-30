from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn import metrics
import pandas as pd
import numpy as np
import sys,io
import xlrd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')

data = xlrd.open_workbook('E:/textClassifyAI/data/0605语料后台数据.xlsx')
table = data.sheets()[0]
nrows = table.nrows

dataList = []
keywordList = []
keywordId = []

def getData():
    for i in range(nrows):
        if i==0:
            continue
        message = table.row_values(i)[0:1][0].strip()
        keyword = table.row_values(i)[2:3][0].strip()
        dataList.append(message)
        keywordList.append(keyword)

# def parseData():
#     newKeyword = list(set(keywordList))
#     dict = {}
#     for item in keywordList:
#         index = newKeyword.index(item)
#         keywordId.append(index)
#         dict[item] = index

def createModel():
    """搜索最优模型"""
    X_train, X_test, y_train, y_test = train_test_split(dataList, keywordList, test_size=0.3)
    # NB
    text_clf = Pipeline([('CV',CountVectorizer()),('TF',TfidfTransformer()),('NB',MultinomialNB())])
    text_clf.fit(X_train,y_train)
    predicted = text_clf.predict(X_test)
    accuracy = np.mean(predicted==y_test)
    print('NB准确率：',accuracy)

    # KNN
    text_clf = Pipeline([('CV',CountVectorizer()),('TF',TfidfTransformer()),('KNN',KNeighborsClassifier())])
    text_clf.fit(X_train,y_train)
    predicted = text_clf.predict(X_test)
    accuracy = np.mean(predicted==y_test)
    print('KNN准确率：',accuracy)

    # SGD
    text_clf = Pipeline([('CV', CountVectorizer()), ('TF', TfidfTransformer()), ('SGD', SGDClassifier())])
    text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_test)
    accuracy = np.mean(predicted == y_test)
    print('SGD准确率：', accuracy)


    # hash
    text_clf = Pipeline([('hash',HashingVectorizer(stop_words='english',non_negative=True,n_features=100000)),('TF', TfidfTransformer()), ('SGD',SGDClassifier())])
    text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_test)
    accuracy = np.mean(predicted == y_test)
    print('hash准确率：', accuracy)

    # SVC
    text_clf = Pipeline([('CV',CountVectorizer()),('TF',TfidfTransformer()),('SVC',LinearSVC())])
    text_clf.fit(X_train,y_train)
    predicted = text_clf.predict(X_test)
    accuracy = np.mean(predicted == y_test)
    print('SVC准确率：', accuracy)

def testGridSearch():
    """搜索最优参数"""
    X_train, X_test, y_train, y_test = train_test_split(dataList, keywordList, test_size=0.3)
    text_clf = Pipeline([('CV',CountVectorizer()),('TF',TfidfTransformer()),('SGD',SGDClassifier())])
    parameters = {
        'CV__ngram_range': [(1, 1), (1, 2)],
        'CV__max_df': (0.1, 0.99),
        # 'CV__min_df': (0.1, 0.99),
        'CV__binary': (True,False),
        'CV__max_features': (None, 5000, 10000),
        'TF__use_idf': (True, False),
        'TF__norm': ('l1', 'l2'),
        'TF__smooth_idf': (True,False),
        'TF__sublinear_tf':(True,False),
        'SGD__alpha': (0.0000001, 0.001),
        'SGD__penalty': ('l2', 'elasticnet'),
        'SGD__max_iter': (5, 1000),
    }
    # GridSearch 寻找最优参数的过程
    flag = 0
    if (flag == 0):
        grid_search = GridSearchCV(text_clf, parameters, n_jobs=1, verbose=1)
        grid_search.fit(X_train, y_train)
        print("Best score: %0.3f" % grid_search.best_score_)
        best_parameters = {}
        best_parameters = grid_search.best_estimator_.get_params()
        print("Out the best parameters")
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

    # 找到最优参数后，利用最优参数训练模型
    # text_clf.set_params(
    #     CV__max_df=0.75,
    #     CV__max_features=None,
    #     CV__ngram_range=(1, 2),
    #     SGD__alpha=1e-05,
    #     SGD__n_iter=10,
    #     TF__use_idf=True)
    # text_clf.fit(X_train, y_train)
    # # 预测
    # pred = text_clf.predict(X_test)
    # # 输出结果
    # accuracy = np.mean(pred == y_test)
    # # print accuracy
    # print("The accuracy of twenty_test is %s" % accuracy)
    #
    # print(metrics.classification_report(y_test, pred, target_names=['all']))
    # array = metrics.confusion_matrix(y_test, pred)
    # print(array)


# def sample_NB():
#     '''
#     提取特征
#     语料文件可以用一个词文档矩阵代表，每行是一个文档，每列是一个标记（即词）。将文档文件转化为数值特征的一般过程被称为向量化。
#     这个特殊的策略（标记，计数和正态化）被称为词袋或者Bag of n-grams表征。用词频描述文档，但是完全忽略词在文档中出现的相对位置信息。
#     参数：
#       stop_words=,指定的停用词；max_df=，超过这个频率的词不作为词典词，默认1.0；min_df=，小于这个频率的词不作为次电磁，默认1（至少出现在一篇文章中）；
#       max_features=，词典最多有多少个词，默认None，如果指定了，则通过词频来排序取舍。vocabulary=，指定的词典。
#     '''
#     CV = CountVectorizer()
#     X_train_bow = CV.fit_transform(text_train.data)
#
#     #TF-IDF计算词的权重
#     TF = TfidfTransformer()
#     X_train_tf = TF.fit_transform(X_train_bow)
#
#     #构建朴素贝叶斯分类器
#     NB = MultinomialNB()
#     clf = NB.fit(X_train_bow,text_train.target)
#
#     #简单测试预测
#     docs_new = ['God is love','OpenGL on the GPU is fast']
#     X_new_bow = CV.transform(docs_new)
#     X_new_tf = TF.transform(X_new_bow)
#     predicted = clf.predict(X_new_tf)
#     for doc,category in zip(docs_new,predicted):
#         print('%r => %s'%(doc,text_train.target_names[category]))


if __name__ == '__main__':
    getData()
    createModel()