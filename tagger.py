"""auto-tagger-lite — 文档自动打标签核心逻辑。

内置标签词表（每个标签一组关键词），结合「关键词命中」与「TF-IDF 相关度」
对一段文档给出多标签。零第三方依赖。
"""
from __future__ import annotations

import math
import re
from collections import Counter

# 内置标签词表：标签 -> 触发关键词（中英混合）
DEFAULT_TAG_VOCAB: dict[str, list[str]] = {
    "科技": ["人工智能", "AI", "算法", "芯片", "编程", "软件", "互联网", "科技",
             "machine learning", "python", "data", "computer"],
    "体育": ["足球", "篮球", "比赛", "运动员", "奥运", "冠军", "联赛", "体育",
             "football", "game", "team"],
    "财经": ["股票", "基金", "汇率", "通胀", "GDP", "央行", "投资", "市场",
             "economy", "stock", "finance"],
    "娱乐": ["电影", "明星", "演唱会", "综艺", "导演", "演员", "娱乐", "music",
             "movie", "star"],
    "美食": ["餐厅", "菜", "火锅", "咖啡", "烘焙", "食谱", "美食", "delicious",
             "food", "restaurant"],
    "健康": ["医院", "医生", "疫苗", "营养", "运动", "睡眠", "健康", "health",
             "medicine"],
    "教育": ["学校", "学生", "考试", "课程", "老师", "大学", "教育", "study",
             "school", "exam"],
    "旅游": ["景点", "酒店", "航班", "攻略", "目的地", "旅游", "travel", "hotel",
             "trip"],
}

_CN_RE = re.compile(r"[\u4e00-\u9fff]")
_EN_RE = re.compile(r"[a-zA-Z]+")


def tokenize(text: str) -> list[str]:
    """极简分词：英文按词切分；中文按 2-gram（相邻汉字对）切分。

    足够在零依赖下支撑关键词命中与 TF-IDF 统计。
    """
    text = text.lower()
    tokens: list[str] = []
    # 英文词
    tokens += _EN_RE.findall(text)
    # 中文连续段切 2-gram
    for seg in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(seg) == 1:
            tokens.append(seg)
        else:
            tokens += [seg[i:i + 2] for i in range(len(seg) - 1)]
    return tokens


def _keyword_hits(text: str, keywords: list[str]) -> list[str]:
    low = text.lower()
    hits = []
    for kw in keywords:
        if kw.lower() in low:
            hits.append(kw)
    return hits


def tag_document(text: str, vocab: dict[str, list[str]] | None = None,
                 top_k: int = 3, min_score: float = 0.0) -> list[dict]:
    """为单篇文档打多标签。

    返回按综合相关度降序排列的标签列表，每项：
      {"tag", "score", "matched"} — matched 为命中的关键词。
    """
    vocab = vocab or DEFAULT_TAG_VOCAB
    scores: list[dict] = []
    for tag, keywords in vocab.items():
        hits = _keyword_hits(text, keywords)
        # 词命中分：命中数量越多、命中词越长（信息量大），分越高
        hit_score = sum(min(len(h), 6) for h in hits)
        if hit_score <= 0:
            continue
        scores.append({"tag": tag, "score": float(hit_score), "matched": hits})
    scores.sort(key=lambda x: x["score"], reverse=True)
    return [s for s in scores if s["score"] >= min_score][:top_k]


def tfidf_rank(query_text: str, documents: list[str],
               top_n: int = 10) -> list[tuple[str, float]]:
    """用 TF-IDF 衡量 query 与文档集合中各词的相关度（辅助排序）。

    返回 (词, tfidf) 降序列表。可用于在已有候选标签内进一步排序。
    """
    # 文档频率
    doc_tokens = [set(tokenize(d)) for d in documents]
    n_docs = max(1, len(doc_tokens))
    df: Counter = Counter()
    for dt in doc_tokens:
        for t in dt:
            df[t] += 1

    q_tokens = tokenize(query_text)
    tf = Counter(q_tokens)
    scored: dict[str, float] = {}
    for term, cnt in tf.items():
        idf = math.log((n_docs + 1) / (df.get(term, 0) + 1)) + 1.0
        scored[term] = (cnt / max(1, len(q_tokens))) * idf
    ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def auto_tag(text: str, documents: list[str] | None = None,
             vocab: dict[str, list[str]] | None = None,
             top_k: int = 3) -> list[str]:
    """对外主接口：返回标签名字符串列表。

    documents 若提供，则用 TF-IDF 对候选标签做二次加权；否则仅用词命中。
    """
    raw = tag_document(text, vocab=vocab, top_k=top_k * 2)
    if not documents:
        return [r["tag"] for r in raw[:top_k]]

    boost = dict(tfidf_rank(text, documents, top_n=50))
    # 对每个标签，用其命中词在 tfidf 中的平均权重做加权
    for r in raw:
        weights = [boost.get(h.lower(), 0.0) for h in r["matched"]]
        r["tfidf_boost"] = sum(weights) / len(weights) if weights else 0.0
    raw.sort(key=lambda x: (x["score"] + x["tfidf_boost"] * 2), reverse=True)
    return [r["tag"] for r in raw[:top_k]]
