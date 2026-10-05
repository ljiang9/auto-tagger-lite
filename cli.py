"""命令行入口：python3 cli.py "一段文档文本" """
import argparse
import json
import sys

from tagger import auto_tag, tag_document, DEFAULT_TAG_VOCAB


def main(argv=None):
    p = argparse.ArgumentParser(description="auto-tagger-lite 文档自动打标签")
    p.add_argument("text", nargs="?", help="待打标签的文档文本")
    p.add_argument("-k", "--top-k", type=int, default=3, help="返回标签个数")
    p.add_argument("--json", action="store_true", help="输出含命中词的详细 JSON")
    args = p.parse_args(argv)

    text = args.text
    if not text:
        text = sys.stdin.read()
    if not text.strip():
        print("错误：未提供文档文本", file=sys.stderr)
        return 2

    if args.json:
        result = tag_document(text, top_k=args.top_k)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        tags = auto_tag(text, top_k=args.top_k)
        print("标签：" + "、".join(tags) if tags else "标签：（无命中）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
