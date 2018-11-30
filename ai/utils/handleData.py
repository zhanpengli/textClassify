from ai import redisClient

def parseData(res):
    dataList = []
    targetList = []
    targetId = []
    targetdict = {}
    for data in res:
        message = data['message']
        target = data['target']
        dataList.append(message)
        targetList.append(target)
    newKeyword = list(set(targetList))
    for item in targetList:
        index = newKeyword.index(item)
        targetId.append(index)
        targetdict[index] = item
    redisClient.set('target',targetdict)
    return dataList,targetId

def handledata(res):
    dataList = []
    targetList = []
    for data in res:
        message = data['message']
        target = data['target']
        dataList.append(message)
        targetList.append(target)
    return dataList,targetList