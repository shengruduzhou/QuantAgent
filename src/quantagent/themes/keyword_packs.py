"""Keyword packs that drive policy → theme mapping.

The ``PolicyIngestor.keyword_to_theme`` map turns crawled policy/news bodies
into one or more theme tags. The themes here align with the **15th Five-Year
Plan** (十五五) and related strategic-industry policy threads issued by the
State Council, NDRC, MIIT, MOST, and Xinhua coverage.

Each entry maps a Chinese/English keyword to a tuple of theme codes. The
theme codes are stable internal identifiers used downstream by the universe
builder and the industry-chain reasoner.
"""

from __future__ import annotations

THEMES_15TH_FIVE_YEAR_PLAN: dict[str, tuple[str, ...]] = {
    # ---- 十五五 / general plan markers --------------------------------------
    "十五五": ("national_15th_five_year_plan",),
    "十四五": ("national_14th_five_year_plan",),
    "中长期规划": ("national_strategic_plan",),
    "国务院": ("national_strategic_plan",),
    # ---- AI / 算力 / 数据要素 -------------------------------------------------
    "ai": ("ai_compute",),
    "人工智能": ("ai_compute",),
    "大模型": ("ai_compute",),
    "算力": ("ai_compute",),
    "数据中心": ("ai_compute",),
    "数据要素": ("data_factor",),
    "数据交易": ("data_factor",),
    # ---- 半导体 / 国产替代 ---------------------------------------------------
    "半导体": ("semiconductor_domestic_substitution",),
    "集成电路": ("semiconductor_domestic_substitution",),
    "光刻": ("semiconductor_domestic_substitution",),
    "国产替代": ("semiconductor_domestic_substitution",),
    "先进制程": ("semiconductor_domestic_substitution",),
    "晶圆": ("semiconductor_domestic_substitution",),
    # ---- 新能源 / 储能 / 电网 -------------------------------------------------
    "储能": ("energy_storage",),
    "新型电力系统": ("energy_storage", "smart_grid"),
    "智能电网": ("smart_grid",),
    "光伏": ("photovoltaic",),
    "风电": ("wind_power",),
    "新能源": ("energy_storage", "new_energy_vehicle"),
    "新能源汽车": ("new_energy_vehicle",),
    "氢能": ("hydrogen_energy",),
    "钠电": ("energy_storage",),
    # ---- 高端装备 / 制造 -----------------------------------------------------
    "高端装备": ("high_end_manufacturing",),
    "工业母机": ("high_end_manufacturing",),
    "机器人": ("humanoid_robotics", "high_end_manufacturing"),
    "人形机器人": ("humanoid_robotics",),
    "智能制造": ("smart_manufacturing", "high_end_manufacturing"),
    # ---- 航空 / 航天 / 低空 --------------------------------------------------
    "商业航天": ("commercial_space",),
    "卫星": ("commercial_space",),
    "低空": ("low_altitude_economy",),
    "低空经济": ("low_altitude_economy",),
    "eVTOL": ("low_altitude_economy",),
    # ---- 生物医药 -------------------------------------------------------------
    "创新药": ("innovative_drug",),
    "生物医药": ("innovative_drug",),
    "中医药": ("traditional_chinese_medicine",),
    "医疗器械": ("medical_devices",),
    # ---- 国防 / 安全 ----------------------------------------------------------
    "军工": ("defense_modernisation",),
    "国防": ("defense_modernisation",),
    "网络安全": ("cyber_security",),
    "信息安全": ("cyber_security",),
    # ---- 农业 / 粮食 ---------------------------------------------------------
    "种业": ("seed_industry",),
    "粮食安全": ("seed_industry", "food_security"),
    # ---- 其他战略产业 --------------------------------------------------------
    "新材料": ("advanced_materials",),
    "稀土": ("rare_earth_strategic",),
    "可控核聚变": ("controlled_fusion",),
    "脑机接口": ("brain_computer_interface",),
}


__all__ = ["THEMES_15TH_FIVE_YEAR_PLAN"]
