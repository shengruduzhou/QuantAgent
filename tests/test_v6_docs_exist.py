from pathlib import Path


def test_v6_chinese_docs_exist_and_are_non_empty():
    docs = [
        "README.md",
        "docs/V6_总览.md",
        "docs/V6_数据与PIT特征.md",
        "docs/V6_模型训练与验证.md",
        "docs/V6_Agent证据系统.md",
        "docs/V6_组合优化与风险控制.md",
        "docs/V6_虚拟交易与历史回放.md",
        "docs/V6_生产可信度与审计.md",
        "docs/V6_运维与故障演练.md",
        "docs/V6_验收标准.md",
    ]
    for path in docs:
        text = Path(path).read_text(encoding="utf-8")
        assert len(text) > 200
        assert "目标" in text

