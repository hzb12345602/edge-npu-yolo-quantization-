# -*- coding: utf-8 -*-
"""
网页正文抓取小工具（用于查官方文档/基准数据）
用法：
  python fetch_text.py <URL> [关键词]      # 有关键词则只打印关键词附近内容
  python fetch_text.py <URL> --links yolo # 打印页面中包含 yolo 的链接
"""
import re
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")


def clean(html):
    t = re.sub(r"<script[\s\S]*?</script>", " ", html)
    t = re.sub(r"<style[\s\S]*?</style>", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"&nbsp;", " ", t)
    return re.sub(r"\s+", " ", t)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    url = sys.argv[1]
    html = fetch(url)

    if len(sys.argv) >= 3 and sys.argv[2] == "--links":
        kw = (sys.argv[3] if len(sys.argv) > 3 else "").lower()
        for l in sorted(set(re.findall(r'href=["\']([^"\']+)["\']', html))):
            if kw in l.lower():
                print(l)
        return

    text = clean(html)
    if len(sys.argv) >= 3:
        kw = sys.argv[2]
        i = text.find(kw)
        if i < 0:
            print(f"[未找到关键词 {kw}] 正文前 1500 字：")
            print(text[:1500])
        else:
            print(text[max(0, i - 200): i + 3000])
    else:
        print(text[:4000])


if __name__ == "__main__":
    main()
