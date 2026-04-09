import os
from collections.abc import Iterable
import re
from difflib import SequenceMatcher


# Common fund company names used as prefixes/suffixes in ETF names.
COMMON_FUND_COMPANIES: tuple[str, ...] = (
    "华夏", "易方达", "华泰柏瑞", "景顺", "景顺长城", "大成", "工银", "工银瑞信", "广发", "华宝",
    "南方", "富国", "天弘", "鹏华", "招商", "国泰", "银华", "万家", "博时", "嘉实",
    "汇添富", "华安", "平安", "建信", "国联安", "前海开源", "东财", "永赢", "摩根", "海富通",
    "兴业", "申万菱信", "中银", "中银证券", "交银", "国联", "融通", "泰康", "浦银", "民生加银",
    "方正富邦", "西部利得", "国寿", "鹏扬", "国投瑞银", "南华", "创金合信", "鑫元", "兴全",
    "弘毅远方", "银河", "汇安", "国寿安保",
)

INDEX_KEYWORDS: tuple[str, ...] = (
    "沪深300", "深证100", "深证50", "上证50", "上证180", "上证380", "中证A500", "中证A50",
    "中证500", "中证800", "中证1000", "中证2000", "创业板50", "创业板200", "创业板",
    "科创50", "科创100", "科创200", "科创", "双创50", "恒生科技", "恒生医疗", "恒生医药",
    "恒生消费", "恒生互联网", "恒生生物科技", "恒生", "港股通", "港股", "红利低波",
    "自由现金流", "现金流", "红利质量", "红利", "消费电子", "食品饮料", "家用电器", "家电",
    "新能源车", "新能源", "电池", "储能", "光伏", "风电", "绿色电力", "电力", "电网设备",
    "人工智能", "机器人", "算力", "云计算", "大数据", "数字经济", "半导体设备", "半导体",
    "芯片", "集成电路", "软件", "通信", "信息技术", "信息安全", "互联网", "医药", "医疗",
    "创新药", "生物科技", "生物医药", "疫苗", "中药", "证券", "券商", "银行", "保险",
    "金融科技", "金融", "消费", "农业", "养殖", "农牧渔", "粮食", "有色金属", "有色",
    "稀有金属", "稀土", "黄金", "油气", "石油天然气", "石油", "煤炭", "化工", "材料",
    "建材", "基建", "工程机械", "高端装备", "航空航天", "通用航空", "军工", "国企", "央企",
    "标普500", "标普油气", "标普消费", "纳斯达克100", "纳指科技", "纳指生物科技", "纳斯达克",
    "纳指", "日经225", "日经", "德国", "法国", "巴西", "沙特", "A500", "A100", "A50",
)

def get_symbol_name_from_fp(fp: str) -> str:
    return os.path.splitext(os.path.basename(fp))[0]


def extract_tracked_index_name(etf_name: str, company_names: Iterable[str] | None = None) -> str:
    """
    Extract tracked index name from ETF display name.

    Supported common patterns:
    1) "{tracked_index}ETF{company_name}"
    2) "{company_name}{tracked_index}ETF"
    """
    name = etf_name.strip().replace("（", "(").replace("）", ")")
    if not name:
        return ""

    if "ETF" not in name:
        return name

    all_companies = {c.strip() for c in COMMON_FUND_COMPANIES}
    if company_names is not None:
        all_companies.update(c.strip() for c in company_names if c and c.strip())
    ordered_companies = sorted(all_companies, key=len, reverse=True)

    def strip_company_edges(text: str) -> str:
        result = text.strip()
        for company in ordered_companies:
            if result.startswith(company):
                result = result[len(company):].strip()
                break
        for company in ordered_companies:
            if result.endswith(company):
                result = result[:-len(company)].strip()
                break
        return result

    etf_pos = name.find("ETF")
    left = name[:etf_pos].strip()
    right = name[etf_pos + 3 :].strip()

    # Pattern: {tracked_index}ETF{company_name}
    if right:
        candidate = strip_company_edges(left)
        return candidate or left

    # Pattern: {company_name}{tracked_index}ETF
    candidate = strip_company_edges(left)
    return candidate or left


def normalize_tracked_index_name(index_name: str) -> str:
    normalized = index_name.strip().replace("（", "(").replace("）", ")")
    normalized = re.sub(r"[()\-_/·\s]+", "", normalized)
    return normalized


def extract_index_tokens(index_name: str) -> set[str]:
    normalized = normalize_tracked_index_name(index_name)
    if not normalized:
        return set()

    tokens: set[str] = set()
    for keyword in INDEX_KEYWORDS:
        if keyword in normalized:
            tokens.add(keyword)

    for token in re.findall(r"[A-Za-z]+\d*|\d+[A-Za-z]*", normalized.upper()):
        tokens.add(token)

    for token in re.findall(r"\d+", normalized):
        tokens.add(token)

    if not tokens:
        tokens.add(normalized)
    return tokens


def tracked_index_similarity(index_name_a: str, index_name_b: str) -> float:
    normalized_a = normalize_tracked_index_name(index_name_a)
    normalized_b = normalize_tracked_index_name(index_name_b)
    if not normalized_a or not normalized_b:
        return 0.0
    if normalized_a == normalized_b:
        return 1.0

    tokens_a = extract_index_tokens(normalized_a)
    tokens_b = extract_index_tokens(normalized_b)
    token_union = tokens_a | tokens_b
    token_intersection = tokens_a & tokens_b
    token_score = len(token_intersection) / len(token_union) if token_union else 0.0

    containment_score = 0.0
    if normalized_a in normalized_b or normalized_b in normalized_a:
        containment_score = min(len(normalized_a), len(normalized_b)) / max(len(normalized_a), len(normalized_b))

    sequence_score = SequenceMatcher(None, normalized_a, normalized_b).ratio()
    return max(token_score, containment_score * 0.9, sequence_score * 0.6)


def tracked_index_distance(index_name_a: str, index_name_b: str) -> float:
    return 1.0 - tracked_index_similarity(index_name_a, index_name_b)
