import time

def timeme(method):
    def wrapper(*args, **kw):
        startTime = round(time.time() * 1000, 3)
        result = method(*args, **kw)
        endTime = round(time.time() * 1000, 3)
        print(endTime - startTime,'ms')
        return result
    return wrapper