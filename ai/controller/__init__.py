from flask import Blueprint

# 创建蓝图对象
api = Blueprint("apps", __name__)

from  . import  receive, recognize, train
