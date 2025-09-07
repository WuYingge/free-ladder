import time
import datetime
from functools import wraps

def intervals(seconds=60):
    time.sleep(seconds)

def retry_with_intervals(max_retries=10, interval_func=None):
    """
    重试装饰器，带有间隔和最大重试次数限制
    
    参数:
    max_retries: 最大重试次数
    interval_func: 间隔函数，在每次重试之间调用
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_time = 0
            while retry_time <= max_retries:
                success = func(*args, **kwargs)
                if success:
                    return success
                
                retry_time += 1
                if retry_time > max_retries:
                    print(f"Never do that again, can't execute {func.__name__} with args {args}")
                    return False
                
                if interval_func:
                    interval_func()
            return False
        return wrapper
    return decorator
    