from sklearn.externals import joblib
from ai import redisClient
from flask import request,jsonify
from . import api
from ai import redisClient
import numpy as np

@api.route('/messenger/recognize', methods=['POST'])
def recognize():
    response = request.get_json()
    message = response['message']
    clf = joblib.load('ai/models/messenger_clf.pkl')
    score = clf.predict_proba(message)
    target = clf.predict(message)
    score_1 = np.ndarray.tolist(score)
    target_1 = np.ndarray.tolist(target)
    score_2 = [max(x) for x in score_1]
    resData = [[x[0],x[1]] for x in zip(target_1,score_2)]
    return jsonify({"code": 200, "data": resData})