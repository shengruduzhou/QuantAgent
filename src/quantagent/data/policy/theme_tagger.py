"""Rule-based theme + sector tagger for policy events.

NLP for Chinese policy documents is its own discipline; this module
ships a deliberately *simple* keyword-rule tagger so the data layer can
ship without a model dependency.  Real teams should replace the rule
table with a fine-tuned classifier, but the rule version is
deterministic, auditable, and good enough for the time-lag model in
Stage 4.2.

Themes are coarse (8 categories): an event can belong to multiple
themes.  ``sectors_hint`` is a soft mapping to sector_level_1 names
used by the rest of the pipeline.

Policy strength bands:
* 1.0 — hard regulation: "暂行规定", "管理办法", "决定", "通知...规定"
* 0.7 — directive: "指导意见", "意见", "指引", "措施"
* 0.4 — soft guidance: "通知", "公告", "答记者问", "答复"
* 0.2 — informational: news, press releases without rule force
"""

from __future__ import annotations

from typing import Any


POLICY_THEMES: tuple[str, ...] = (
    "monetary",      # 货币政策、利率、降准、流动性
    "fiscal",        # 财税、减税、专项债
    "regulation",    # 监管、合规、IPO、退市
    "industry",      # 产业政策、补贴、扶持
    "consumption",   # 消费、内需
    "real_estate",   # 房地产、土地、保障房
    "tech_innovation",  # 科创、半导体、人工智能、新能源
    "open_economy",  # 对外开放、自贸区、外资
)


THEME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "monetary": (
        "利率", "存款准备金", "降准", "降息", "LPR", "公开市场", "MLF",
        "再贷款", "再贴现", "货币政策", "央行", "流动性",
    ),
    "fiscal": (
        "减税", "退税", "税收", "专项债", "国债", "财政", "财税",
        "财政部", "增值税", "企业所得税", "印花税",
    ),
    "regulation": (
        "监管", "证监会", "退市", "IPO", "再融资", "信息披露", "上市公司",
        "并购重组", "证券法", "违规", "处罚", "立案",
    ),
    "industry": (
        "产业政策", "补贴", "扶持", "鼓励", "战略性新兴",
        "高质量发展", "制造业", "工业", "扶贫",
    ),
    "consumption": (
        "消费", "内需", "以旧换新", "购车", "购房补贴", "消费券",
        "促消费", "扩内需",
    ),
    "real_estate": (
        "房地产", "住房", "土地", "保障房", "限购", "限贷", "棚改",
        "公积金", "二手房", "新房",
    ),
    "tech_innovation": (
        "科创", "科技创新", "半导体", "集成电路", "人工智能", "新能源",
        "新材料", "生物医药", "数字经济", "5G", "6G", "量子",
    ),
    "open_economy": (
        "对外开放", "自贸区", "外资", "QFII", "RQFII", "陆股通",
        "互联互通", "一带一路",
    ),
}


# Keyword → industry label mapping. Keys are 申万一级 (Shenwan Level-1)
# sector names so ``sectors_hint`` joins directly onto the
# ``silver/sector_map`` ``sector_level_1`` column used across the pipeline
# (the hybrid sector-pool merge keys on this exact vocabulary). An event can
# match multiple sectors.
SECTOR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "银行": ("银行", "信贷", "存款", "贷款", "理财"),
    "非银金融": ("保险", "再保险", "保单", "证券", "券商", "投行", "经纪", "信托"),
    "房地产": ("房地产", "住房", "保障房", "棚改", "公积金", "限购", "限贷", "二手房", "新房", "土地出让"),
    "建筑装饰": ("基建", "建筑", "工程", "中铁", "中建", "城市更新", "城镇化"),
    "建筑材料": ("水泥", "玻璃", "建材", "陶瓷"),
    "电子": ("半导体", "集成电路", "芯片", "晶圆", "光刻", "消费电子", "面板", "封测"),
    "计算机": ("数字经济", "人工智能", "云计算", "大数据", "软件", "信创", "国产化", "工业互联网", "网络安全"),
    "通信": ("电信", "5G", "6G", "运营商", "光通信", "光模块", "卫星互联网"),
    "医药生物": ("医药", "生物医药", "中药", "创新药", "医疗器械", "疫苗", "医疗", "集采"),
    "电力设备": ("光伏", "风电", "储能", "动力电池", "锂电", "充电桩", "特高压", "电网"),
    "公用事业": ("电力", "水务", "燃气", "核电", "供热"),
    "石油石化": ("石油", "天然气", "原油", "炼化", "油气"),
    "煤炭": ("煤炭", "焦煤", "动力煤", "煤化工"),
    "基础化工": ("化工", "化学", "农药", "化肥", "氟化工"),
    "有色金属": ("有色", "稀土", "锂矿", "黄金", "电解铝"),
    "钢铁": ("钢铁", "黑色金属", "铁矿石", "特钢"),
    "汽车": ("汽车", "新能源汽车", "智能驾驶", "整车", "汽车零部件", "以旧换新"),
    "机械设备": ("机械", "装备", "机床", "工程机械", "机器人", "数控"),
    "国防军工": ("军工", "国防", "航空", "航天", "兵器", "导弹", "商业航天"),
    "农林牧渔": ("农业", "农村", "种业", "养殖", "畜牧", "渔业", "粮食", "乡村振兴"),
    "食品饮料": ("食品", "饮料", "白酒", "啤酒", "乳制品", "调味品"),
    "商贸零售": ("零售", "商超", "百货", "电商", "免税"),
    "社会服务": ("旅游", "酒店", "教育", "景区", "人力资源"),
    "纺织服饰": ("纺织", "服装", "服饰", "品牌服饰"),
    "轻工制造": ("轻工", "造纸", "包装", "家具"),
    "家用电器": ("家电", "空调", "冰箱", "白电", "厨电"),
    "传媒": ("传媒", "影视", "游戏", "广告", "出版", "院线", "短视频"),
    "环保": ("环保", "污水", "固废", "碳中和", "节能", "环境治理"),
    "交通运输": ("交通运输", "物流", "航运", "港口", "铁路", "高速公路", "快递"),
    "美容护理": ("美容", "化妆品", "个护"),
}


# Phrases that mark the regulatory hardness band.
STRENGTH_BANDS: tuple[tuple[float, tuple[str, ...]], ...] = (
    (1.0, ("规定", "管理办法", "决定", "暂行规定", "条例", "法")),
    (0.7, ("指导意见", "意见", "指引", "措施", "实施细则")),
    (0.4, ("通知", "公告", "答记者问", "答复", "公开征求意见")),
)


def _keyword_hits(text: str, keywords: tuple[str, ...]) -> tuple[str, ...]:
    if not text:
        return ()
    return tuple(kw for kw in keywords if kw in text)


def tag_policy_event(
    title: str,
    body: str = "",
    *,
    themes: tuple[str, ...] = POLICY_THEMES,
) -> dict[str, Any]:
    """Tag a single policy event with themes, sector hints, and strength.

    Returns a dict shaped to merge into a policy_events row:
    ``themes``, ``sectors_hint`` (lists), ``policy_strength`` (float).
    Empty/None inputs return zero-valued tags so callers can ingest
    rows that lack body summaries without crashing.
    """
    title = title or ""
    body = body or ""
    combined = f"{title}\n{body}"

    tagged_themes: list[str] = []
    for theme in themes:
        kw = THEME_KEYWORDS.get(theme, ())
        if _keyword_hits(combined, kw):
            tagged_themes.append(theme)

    tagged_sectors: list[str] = []
    for sector, kw in SECTOR_KEYWORDS.items():
        if _keyword_hits(combined, kw):
            tagged_sectors.append(sector)

    strength = 0.2  # informational by default
    # Bands are sorted strong-first; first hit wins.
    for band_value, band_kws in STRENGTH_BANDS:
        if _keyword_hits(title + body, band_kws):
            strength = band_value
            break

    return {
        "themes": tagged_themes,
        "sectors_hint": tagged_sectors,
        "policy_strength": float(strength),
    }
