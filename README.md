# auto-tagger-lite

文档自动打标签小工具。内置 8 类标签词表（科技 / 体育 / 财经 / 娱乐 / 美食 / 健康 / 教育 / 旅游），
先按关键词命中打分，再可选地用 **TF-IDF** 相关度二次加权，给一段文档打出多个标签。

纯 Python 3.10+，**零第三方依赖**。

## 功能

- 内置标签词表，中英混合关键词命中；
- 中文按 2-gram、英文按词切分（零依赖分词）；
- 关键词命中数 + 命中词信息量打分；
- 提供文档集合时，用 TF-IDF 对候选标签二次排序；
- 可自定义标签词表。

## 快速开始

```bash
python3 cli.py "这家公司发布了新的人工智能芯片和机器学习算法"
# 标签：科技、财经
```

## 使用示例

```bash
# 详细模式（输出命中的关键词）
python3 cli.py --json "央行降息，股票市场波动"

# 控制返回标签个数
python3 cli.py -k 2 "足球比赛明星云集，同时有科技直播技术"

# 从管道读取
echo "电影明星的新作品上映" | python3 cli.py
```

## 无 API Key 如何运行

本工具**完全不需要任何 API Key**，所有功能均为本地规则 + TF-IDF 计算，开箱即用。

## 目录结构

```
auto-tagger-lite/
├── tagger.py     # 核心：词表、分词、命中打分、TF-IDF
├── cli.py        # 命令行入口
├── tests/
│   └── test_tagger.py
├── README.md
├── LICENSE       # MIT
└── .gitignore
```

## 测试

```bash
python3 -m unittest discover -s tests
```

## 许可证

[MIT](./LICENSE)
