from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn import metrics
import numpy as np
import sys,io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')

def searchModel(data,target):
    """搜索最优模型"""
    X_train, X_test, y_train, y_test = train_test_split(data, np.array(target), test_size=0.3)
    # NB
    text_clf = Pipeline([('CV',CountVectorizer()),('TF',TfidfTransformer()),('NB',MultinomialNB())])
    text_clf.fit(X_train,y_train)
    predicted = text_clf.predict(X_test)
    accuracy = np.mean(predicted==y_test)
    print('NB准确率：',accuracy)
    joblib.dump(text_clf,'E:/cfAi/data/text_clf.pkl')

    # KNN
    text_clf = Pipeline([('CV',CountVectorizer()),('TF',TfidfTransformer()),('KNN',KNeighborsClassifier())])
    text_clf.fit(X_train,y_train)
    predicted = text_clf.predict(X_test)
    accuracy = np.mean(predicted==y_test)
    print('KNN准确率：',accuracy)

    # # SGD
    text_clf = Pipeline([('CV', CountVectorizer()), ('TF', TfidfTransformer()), ('SGD', SGDClassifier())])
    text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_test)
    accuracy = np.mean(predicted == y_test)
    print('SGD准确率：', accuracy)

    # # hash
    text_clf = Pipeline([('hash',HashingVectorizer(stop_words='english',non_negative=True,n_features=100000)),('TF', TfidfTransformer()), ('SGD',SGDClassifier())])
    text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_test)
    accuracy = np.mean(predicted == y_test)
    print('hash准确率：', accuracy)

def testGridSearch(data,target):
    """搜索最优参数"""
    X_train, X_test, y_train, y_test = train_test_split(data, np.array(target), test_size=0.3)
    text_clf = Pipeline([('CV',CountVectorizer()),('TF',TfidfTransformer()),('SGD',SGDClassifier())])
    parameters = {
        'CV__ngram_range': [(1, 1), (1, 2)],
        'CV__max_df': (0.5, 0.75),
        'CV__max_features': (None, 5000, 10000),
        'TF__use_idf': (True, False),
        #  'TF__norm': ('l1', 'l2'),
        'SGD__alpha': (0.00001, 0.000001),
        #  'SGD__penalty': ('l2', 'elasticnet'),
        'SGD__n_iter': (10, 50),
    }
    # GridSearch 寻找最优参数的过程
    flag = 0
    if (flag != 0):
        grid_search = GridSearchCV(text_clf, parameters, n_jobs=1, verbose=1)
        grid_search.fit(X_train, y_train)
        print("Best score: %0.3f" % grid_search.best_score_)
        best_parameters = {}
        best_parameters = grid_search.best_estimator_.get_params()
        print("Out the best parameters")
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

    # 找到最优参数后，利用最优参数训练模型
    text_clf.set_params(
        CV__max_df=0.75,
        CV__max_features=None,
        CV__ngram_range=(1, 2),
        SGD__alpha=1e-05,
        SGD__n_iter=10,
        TF__use_idf=True)
    text_clf.fit(X_train, y_train)
    # 预测
    pred = text_clf.predict(X_test)
    # 输出结果
    accuracy = np.mean(pred == y_test)
    # print accuracy
    print("The accuracy of twenty_test is %s" % accuracy)
    print(metrics.classification_report(y_test, pred, target_names=['all']))
    array = metrics.confusion_matrix(y_test, pred)
    print(array)

# def createModel(data,target):
#     SGD_clf = Pipeline([('CV', CountVectorizer()), ('TF', TfidfTransformer()), ('SVC', SVC())])
#     SGD_clf.set_params(
#         CV__max_df = 0.75,
#         CV__max_features = None,
#         CV__ngram_range = (1, 2),
#         SVC__probability=True,
#         TF__use_idf = True
#     )
#     SGD_clf.fit(data, target)
#     joblib.dump(SGD_clf, 'ai/models/SGD_clf.pkl')

def createModel(data,target):
    messenger_clf = Pipeline([('CV', CountVectorizer()), ('TF', TfidfTransformer()), ('SVC', SVC())])
    messenger_clf.set_params(
        CV__max_df = 0.5,
        CV__max_features = None,
        CV__ngram_range = (1, 1),
        # SGD__alpha = 1e-05,
        # SGD__n_iter = 10,
        # SGD__loss= 'log',
        TF__use_idf = True,
        SVC__kernel='linear',
        SVC__probability=True
    )
    messenger_clf.fit(data, target)
    joblib.dump(messenger_clf, 'ai/models/messenger_clf.pkl')

if __name__ == '__main__':
    pass