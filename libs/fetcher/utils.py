import pandas as pd
from typing import Dict, List, Tuple
import requests
import math
from datetime import datetime, timedelta
from proxy.proxy import get_proxy
from utils.interval_utils import intervals

def fetch_paginated_data(url: str, base_params: Dict, timeout: int = 15):
    """
    东方财富-分页获取数据并合并结果
    https://quote.eastmoney.com/f1.html?newcode=0.000001
    :param url: 股票代码
    :type url: str
    :param base_params: 基础请求参数
    :type base_params: dict
    :param timeout: 请求超时时间
    :type timeout: str
    :return: 合并后的数据
    :rtype: pandas.DataFrame
    """
    # 复制参数以避免修改原始参数
    params = base_params.copy()
    # 获取第一页数据，用于确定分页信息
    r = requests.get(url, params=params, timeout=timeout, proxies=get_proxy())
    data_json = r.json()
    # 计算分页信息
    per_page_num = len(data_json["data"]["diff"])
    total_page = math.ceil(data_json["data"]["total"] / per_page_num)
    # 存储所有页面数据
    temp_list = []
    # 添加第一页数据
    temp_list.append(pd.DataFrame(data_json["data"]["diff"]))
    # 获取进度条
    # 获取剩余页面数据
    for page in range(2, total_page + 1):
        intervals(0.5)
        params.update({"pn": page})
        r = requests.get(url, params=params, timeout=timeout, proxies=get_proxy())
        data_json = r.json()
        inner_temp_df = pd.DataFrame(data_json["data"]["diff"])
        temp_list.append(inner_temp_df)
    # 合并所有数据
    temp_df = pd.concat(temp_list, ignore_index=True)
    temp_df["f3"] = pd.to_numeric(temp_df["f3"], errors="coerce")
    temp_df.sort_values(by=["f3"], ascending=False, inplace=True, ignore_index=True)
    temp_df.reset_index(inplace=True)
    temp_df["index"] = temp_df["index"].astype(int) + 1
    return temp_df


def generate_time_slices_alternative(
    days_back: int, 
    max_slice_days: int = 50
) -> List[Tuple[str, str]]:
    """
    使用range函数的替代实现，更简洁
    
    Args:
        days_back: 需要回溯的总天数
        max_slice_days: 每个切片的最大天数，默认为50
    
    Returns:
        时间切片列表，每个切片为(start_date_str, end_date_str)格式
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back - 1)
    
    slices = []
    
    # 使用range函数按最大切片天数步进
    for i in range(0, days_back, max_slice_days):
        # 当前切片的开始日期
        slice_start = start_date + timedelta(days=i)
        
        # 当前切片的结束日期
        slice_end = slice_start + timedelta(days=min(max_slice_days, days_back - i) - 1)
        
        # 确保不超过总结束日期
        if slice_end > end_date:
            slice_end = end_date
        
        start_str = slice_start.strftime("%Y%m%d")
        end_str = slice_end.strftime("%Y%m%d")
        
        slices.append((start_str, end_str))
    
    return slices
