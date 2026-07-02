"""Stage 10.2b — 主营收入 / 概念纯度 (no market-cap proxy).

Verifies how much of a company's business is *actually* the concept, from its
主营业务构成 (东财 stock_zygc_em, revenue-share breakdown) matched against the
concept's keyword set. Emits hard, evidenced fields:

  revenue_exposure_pct   sum of 收入比例 for 主营 segments matching the concept (0..1)
  concept_purity_source  revenue_breakdown | main_business_text | unknown
  purity_confidence      high(>=30%) | medium(10-30%) | low(<10%) | none
  last_verified_date     YYYY-MM-DD of the 主营 report used

When there is no evidence the value is `unknown` — never fabricated, never a
market-cap proxy. Network fetch is guarded (fail-soft): returns unknown on
throttle/failure so the daily scan never breaks.
"""
from __future__ import annotations

import pandas as pd

# concept board -> keyword set for matching 主营构成 text. Board name itself is
# always included; synonyms curated + MINED from real 主营构成 segment vocabulary
# (provenance: runtime/stage10_concept/business_vocab_mined.csv — each added term
# recurs in real member 主营 with its example stocks). Unlisted boards fall back
# to the board-name token. Pharma service-companies (CRO/创新药) intentionally kept
# minimal — segment naming (化学/测试业务) is not concept-matchable -> honest unknown.
CONCEPT_KEYWORDS: dict[str, tuple[str, ...]] = {
    # --- semiconductor / storage (mined: 半导体器件, 集成电路材料, CMP, 存储测试 ...) ---
    "高带宽内存": ("HBM", "高带宽", "存储", "内存", "DRAM", "封装基板", "存储芯片",
                "半导体存储", "存储器件", "半导体存储器件测试", "环氧塑封", "特种集成电路"),
    "存储芯片": ("存储", "DRAM", "NAND", "NOR", "内存", "闪存", "存储芯片", "存储器",
              "主控", "存储器件", "集成电路材料", "被动电子元器件"),
    "CPO概念": ("光模块", "CPO", "硅光", "光引擎", "光器件", "光通信", "光电子器件", "光收发模块"),
    "光通信模块": ("光模块", "光通信", "光器件", "光收发", "光引擎", "光通信器件", "光电子器件"),
    "铜缆高速连接": ("高速连接", "铜缆", "连接器", "高速线", "背板连接"),
    "液冷概念": ("液冷", "冷板", "浸没", "散热", "温控", "CDU"),
    "先进封装": ("先进封装", "封装", "Chiplet", "倒装", "晶圆级封装", "封测",
              "封装测试", "晶圆测试", "集成电路封装"),
    "玻璃基板": ("玻璃基板", "载板", "封装基板", "TGV"),
    "PCB": ("PCB", "印制电路", "线路板", "覆铜板", "载板"),
    "MLCC": ("MLCC", "电容", "被动元件", "陶瓷电容", "电子元器件", "被动电子元器件"),
    "被动元件概念": ("被动元件", "电容", "电感", "电阻", "MLCC", "被动电子元器件"),
    "中芯概念": ("晶圆", "代工", "半导体", "刻蚀", "薄膜", "光刻", "芯片制造", "半导体器件",
              "集成电路", "化学机械抛光", "CMP", "功率半导体", "功率半导体芯片"),
    "半导体概念": ("半导体", "芯片", "集成电路", "晶圆", "封测", "半导体器件", "封装测试"),
    "光刻机(胶)": ("光刻胶", "光刻", "光刻机", "电子化学品", "湿电子", "化学机械抛光",
                "集成电路材料", "光学镜头", "合成树脂", "有机高分子改性材料"),
    "第三代半导体": ("碳化硅", "SiC", "氮化镓", "GaN", "第三代半导体", "功率器件", "功率半导体器件"),
    "第四代半导体": ("氧化镓", "金刚石", "第四代半导体", "超宽禁带"),
    "碳化硅": ("碳化硅", "SiC", "衬底", "外延", "半导体器件", "功率半导体器件", "功率器件", "封装测试"),
    "氮化镓": ("氮化镓", "GaN", "射频", "功率", "功率器件"),
    "IGBT概念": ("IGBT", "功率半导体", "功率模块", "MOSFET", "功率器件"),
    "汽车芯片": ("车规", "汽车芯片", "MCU", "车载", "功率", "车规级芯片", "模拟芯片"),
    # --- industrial gases (mined: 电子大宗气体, 空分设备, 氧气/氮气, 制冷剂 ...) ---
    "氦气概念": ("氦气", "工业气体", "特种气体", "电子特气", "气体", "电子大宗气体",
              "空分设备", "氧气", "氮气", "通用工业气体"),
    "工业气体": ("工业气体", "空分", "特种气体", "电子特气", "氢气", "氦气", "电子大宗气体",
              "空分设备", "制冷剂", "氧气", "氮气", "通用工业气体"),
    "CRO": ("CRO", "CDMO", "CXO", "药物研发", "临床", "外包"),
    "创新药": ("创新药", "新药", "单抗", "双抗", "ADC", "license", "管线"),
    "单抗概念": ("单抗", "抗体", "生物药", "ADC"),
    "CAR-T细胞疗法": ("CAR-T", "细胞治疗", "免疫细胞", "TCR"),
    "减肥药": ("GLP-1", "减重", "司美格鲁肽", "减肥", "降糖"),
    "重组蛋白": ("重组蛋白", "蛋白", "酶", "生物制品"),
    "人形机器人": ("机器人", "人形", "灵巧手", "执行器", "关节"),
    "减速器": ("减速器", "谐波", "RV减速", "丝杠", "传动"),
    "机器人执行器": ("执行器", "电机", "空心杯", "丝杠", "关节模组"),
    "固态电池": ("固态电池", "固态电解质", "硫化物", "氧化物", "锂金属"),
    "钠离子电池": ("钠离子", "钠电", "普鲁士蓝", "硬碳"),
    "复合集流体": ("复合集流体", "复合铜箔", "复合铝箔", "PET铜箔"),
    "锂矿概念": ("锂矿", "锂盐", "碳酸锂", "氢氧化锂", "锂云母"),
    "商业航天": ("商业航天", "火箭", "卫星", "运载", "发动机"),
    "卫星互联网": ("卫星", "星座", "低轨", "通信卫星", "地面站"),
    "碳纤维": ("碳纤维", "碳纤", "复合材料", "预浸料"),
    "稀土永磁": ("稀土", "永磁", "钕铁硼", "磁材"),
    "黄金概念": ("黄金", "金矿", "贵金属", "采金"),
    "PEEK材料概念": ("PEEK", "聚醚醚酮", "特种工程塑料"),
}

_PCT_COLS = ("收入比例", "占主营业务收入比例", "比例", "营收占比")


def keywords_for(board: str) -> tuple[str, ...]:
    base = CONCEPT_KEYWORDS.get(board, ())
    # always include a cleaned board token
    token = board.replace("概念", "").replace("(", "").replace(")", "").replace("（", "").replace("）", "")
    return tuple(dict.fromkeys((*base, token)))   # dedup, preserve order


def revenue_exposure(zygc: pd.DataFrame, board: str) -> tuple[float, str]:
    """(revenue_exposure_pct in 0..1, report_date) from a 主营构成 frame; (nan,'') if none."""
    if zygc is None or zygc.empty:
        return float("nan"), ""
    kws = keywords_for(board)
    # NB: must NOT match "分类类型" (the category-type col) — segment names live in 主营构成
    seg_col = (next((c for c in zygc.columns if "主营构成" in c), None)
               or next((c for c in zygc.columns if "项目" in c or "构成" in c), None))
    pct_col = next((c for c in zygc.columns if any(p in c for p in _PCT_COLS)), None)
    date_col = next((c for c in zygc.columns if "报告期" in c or "日期" in c), None)
    if seg_col is None or pct_col is None:
        return float("nan"), ""
    df = zygc.copy()
    # pick ONE breakdown view (按产品分类 preferred, else 按行业分类) to avoid
    # triple-counting across 产品/行业/地区, and the latest report period only.
    if "分类类型" in df.columns:
        for view in ("按产品分类", "按产品", "按行业分类", "按行业"):
            if (df["分类类型"].astype(str) == view).any():
                df = df[df["分类类型"].astype(str) == view]
                break
    if date_col is not None:
        latest = df[date_col].astype(str).max()
        df = df[df[date_col].astype(str) == latest]
        rdate = str(latest)[:10]
    else:
        rdate = ""
    pct = pd.to_numeric(df[pct_col].astype(str).str.replace("%", "").str.replace(",", ""), errors="coerce")
    if pct.max() is not None and pct.max() > 1.5:   # percentages 0..100 -> 0..1
        pct = pct / 100.0
    mask = df[seg_col].astype(str).apply(lambda t: any(k and k in t for k in kws))
    exposure = float(pct[mask].clip(lower=0).sum())
    return (min(exposure, 1.0) if mask.any() else 0.0), rdate


def purity_record(zygc: pd.DataFrame | None, board: str, *, asof: str) -> dict:
    """Hard, evidenced purity fields; `unknown` when no evidence."""
    exp, rdate = revenue_exposure(zygc, board) if zygc is not None else (float("nan"), "")
    if zygc is None or zygc.empty or pd.isna(exp):
        return {"revenue_exposure_pct": None, "concept_purity_source": "unknown",
                "purity_confidence": "none", "last_verified_date": None}
    conf = "high" if exp >= 0.30 else "medium" if exp >= 0.10 else "low" if exp > 0 else "none"
    return {"revenue_exposure_pct": round(exp, 3), "concept_purity_source": "revenue_breakdown",
            "purity_confidence": conf, "last_verified_date": rdate or asof}


import os
from pathlib import Path

_ZYGC_CACHE = Path("runtime/stage10_concept/raw/zygc")


def fetch_zygc(code: str, *, allow_network: bool = False) -> pd.DataFrame | None:
    """主营构成 (东财). Reuses the mined per-stock cache first; guarded network —
    returns None on disabled/throttled network."""
    f = _ZYGC_CACHE / f"{code.split('.')[0].zfill(6)}.parquet"
    if f.exists():
        try:
            return pd.read_parquet(f)
        except Exception:
            pass
    if not allow_network:
        return None
    try:
        import akshare as ak
        df = ak.stock_zygc_em(symbol=_em_symbol(code))
        if df is not None and not df.empty:
            f.parent.mkdir(parents=True, exist_ok=True)
            df.astype({c: str for c in df.columns if df[c].dtype == object}).to_parquet(f)
        return df
    except Exception:
        return None


def _em_symbol(code: str) -> str:
    c = code.split(".")[0]
    return ("SH" + c) if c[0] in "6" else ("BJ" + c) if c[0] in "489" else ("SZ" + c)
