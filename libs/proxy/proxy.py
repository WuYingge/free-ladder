import requests
import time
import hashlib
import os
import dotenv
import json
from random import choice, randint
from typing import Optional


dotenv.load_dotenv()

# 提取订单
"""
    orderId:提取订单号
    secret:用户密钥
    num:提取IP个数
    pid:省份
    cid:城市
    type：请求类型，1=http/https,2=socks5
    unbindTime:使用时长，秒/s为单位
    noDuplicate:去重，0=不去重，1=去重
    lineSeparator:分隔符
"""
class ProxyAccount:
    orderId: str = os.getenv("PROXY_ORDER_ID") or ""
    secret: str = os.getenv("PROXY_SECRET") or ""
    num = "200"
    pid = "-1"
    cid = ""
    type = "1"
    noDuplicate = "0"
    lineSeparator = "0"
    
    def __init__(self) -> None:
        if not self.orderId or not self.secret:
            raise EnvironmentError("no order id or no secret for proxy")


class ProxyPool:
    
    def __init__(self):
        self.unbindTime = 600
        self.start_time = int(time.time()) #时间戳
        self.pool = self._get_proxies()
    
    # 计算sign
    def _get_proxies(self):
        txt = "orderId=" + ProxyAccount.orderId + "&" + "secret=" + ProxyAccount.secret + "&" + "time=" + str(self.start_time) # type: ignore
        sign = hashlib.md5(txt.encode()).hexdigest()
        # 访问URL获取IP
        url = "http://api.hailiangip.com:8422/api/getIp?type=1" + "&num=" + ProxyAccount.num + "&pid=" + ProxyAccount.pid + "&unbindTime=" + str(self.unbindTime) + "&cid=" + ProxyAccount.cid +  "&orderId=" + ProxyAccount.orderId + "&time=" + str(self.start_time) + "&sign=" + sign + "&dataType=0" + "&lineSeparator=" + ProxyAccount.lineSeparator + "&noDuplicate=" + ProxyAccount.noDuplicate
        my_response = requests.get(url).content
        js_res = json.loads(my_response)
        return [
            {'http': f"http://{dic['ip']}:{dic['port']}","https": f"http://{dic['ip']}:{dic['port']}"}
            for dic in js_res["data"]
        ]
        
    def get_proxy(self):
        if self._check_timeout() or not self.pool:
            self.refresh()
        return self.pool.pop(randint(0, len(self.pool)-1))
    
    def _check_timeout(self):
        return int(time.time()) - self.start_time - 10 > self.unbindTime
    
    def refresh(self):
        self.start_time = int(time.time())
        self.pool = self._get_proxies()
        
PROXY_POOL = ProxyPool()
def get_proxy():
    return PROXY_POOL.get_proxy()
