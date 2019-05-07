import datetime, time

def stringify(strs):
    if isinstance(strs, list):
        return ["'" + str(a) + "'" for a in strs]
    else:
        return "'" + strs + "'"

def timeme(method):
    def wrapper(*args, **kw):
        startTime = round(time.time() * 1000, 3)
        result = method(*args, **kw)
        endTime = round(time.time() * 1000, 3)
        print((endTime - startTime) / 1000, ' seconds')
        return result
    return wrapper