import time
def timestamp():
    now = time.strftime('%y%m%d_%H%M%S', time.localtime(time.time()))
    return now
