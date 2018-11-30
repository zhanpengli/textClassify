# import flask_mongoengine
# from ai import db
# import datetime
#
# """连接mongodb，可多个"""
#     MONGODB_SETTINGS = [
#         {
#             "ALIAS": "ai_db",
#             "DB": "ai_db",
#             "HOST": "127.0.0.1",
#             "PORT": 27017
#         },
#         {
#             "ALIAS": "ai_key",
#             "DB": "ai_key",
#             "HOST": "127.0.0.1",
#             "PORT": 27017
#         },
#     ]
#
#
# class baseModel(object):
#     """模型基类"""
#     create_time = db.DateTimeField(default=datetime.datetime.utcnow)
#     update_time = db.DateTimeField(default=datetime.datetime.utcnow)
#
# class ai_data(baseModel,db.Document):
#     message = db.StringField(required=True)
#     target = db.StringField(required=True)
#     meta = {'db_alias': 'ai_db'}
