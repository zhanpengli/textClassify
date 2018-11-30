from flask import request,jsonify
from . import api
from ai import mongoClient
# from ai.controller.train import scheduler
from ai.controller.train import trainModel

@api.route('/messenger/add', methods=['POST'])
def messageAdd():
    response = request.get_json()
    message = response['message']
    db = mongoClient.ai_db
    db.ai_data.insert_many(message)
    res = trainModel()
    if res['code'] == 200:
        return jsonify({"code":200, "msg":"success"})
    else:
        return jsonify({"code":400,'msg':'fail'})