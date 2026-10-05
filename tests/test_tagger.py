"""auto-tagger-lite 单元测试。运行：python3 -m unittest discover -s tests"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tagger import auto_tag, tag_document, tokenize, tfidf_rank  # noqa: E402


class TestTokenize(unittest.TestCase):
    def test_mixed(self):
        toks = tokenize("我喜欢Python编程")
        self.assertIn("python", toks)
        self.assertTrue(any(t.startswith("我喜") for t in toks))


class TestTagger(unittest.TestCase):
    def test_tech_doc(self):
        tags = auto_tag("这家公司发布了新的人工智能芯片和机器学习算法", top_k=2)
        self.assertIn("科技", tags)

    def test_finance_doc(self):
        tags = auto_tag("央行降息，股票市场与汇率双双波动，投资情绪谨慎", top_k=2)
        self.assertIn("财经", tags)

    def test_detail_has_matched(self):
        res = tag_document("电影明星在导演的新作品中表现出色", top_k=1)
        self.assertEqual(res[0]["tag"], "娱乐")
        self.assertTrue(len(res[0]["matched"]) >= 1)

    def test_no_hit(self):
        self.assertEqual(auto_tag("今天天气很好", top_k=3), [])

    def test_top_k_respected(self):
        text = "足球比赛明星云集，同时涉及财经赞助与科技直播技术"
        tags = auto_tag(text, top_k=2)
        self.assertLessEqual(len(tags), 2)


class TestTfidf(unittest.TestCase):
    def test_ranks_query(self):
        docs = ["人工智能芯片", "足球比赛", "人工智能算法"]
        ranked = tfidf_rank("人工智能", docs, top_n=3)
        terms = [t for t, _ in ranked]
        self.assertTrue(any("人工" in t for t in terms))


if __name__ == "__main__":
    unittest.main()
