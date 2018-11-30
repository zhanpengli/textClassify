from flask import request,jsonify
from ai.controller import api
from ai import mongoClient
from ai.utils.trainUtils import createModel
from ai.utils.handleData import parseData,handledata
from apscheduler.schedulers.blocking import BlockingScheduler

def trainModel():
    db = mongoClient.ai_db
    results = db.ai_data.find()
    if results:
        data, target= handledata(results)
        createModel(data, target)
        return {"code":200}

# scheduler = BlockingScheduler()
# scheduler.add_job(trainModel, 'interval', seconds = 10)
