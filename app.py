import os
import re
import html
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Optional, List, Tuple

import requests
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent


# =========================
# 基本設定
# =========================
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
PORT = int(os.getenv("PORT", "10000"))

if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    raise ValueError("請先設定 LINE_CHANNEL_ACCESS_TOKEN 和 LINE_CHANNEL_SECRET 環境變數")

app = Flask(__name__)

configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

BASE_URL = "https://line-stock-bot-jwjp.onrender.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

# 簡單股票名稱對照
STOCK_MAP = {
    "台積電": "2330",
    "2330": "2330",
    "聯電": "2303",
    "2303": "2303",
    "鴻海": "2317",
    "2317": "2317",
    "金寶": "2312",
    "2312": "2312",
    "仁寶": "2324",
    "2324": "2324",
    "廣達": "2382",
    "2382": "2382",
    "華碩": "2357",
    "2357": "2357",
    "技嘉": "2376",
    "2376": "2376",
    "緯創": "3231",
    "3231": "3231",
    "英業達": "2356",
    "2356": "2356",
    "群創": "3481",
    "3481": "3481",
    "友達": "2409",
    "2409": "2409",
    "中鋼": "2002",
    "2002": "2002",
    "長榮": "2603",
    "2603": "2603",
    "陽明": "2609",
    "2609": "2609",
    "萬海": "2615",
    "2615": "2615",
    "國泰金": "2882",
    "2882": "2882",
    "富邦金": "2881",
    "2881": "2881",
    "元大台灣50": "0050",
    "0050": "0050",
}


# =========================
# 工具函式
# =========================
def normalize_stock_query(text: str) -> str:
    text = text.strip()
    text = text.replace(".TW", "").replace(".tw", "")
    return text


def resolve_stock_code(keyword: str) -> Optional[str]:
    keyword = normalize_stock_query(keyword)
    if keyword in STOCK_MAP:
        return STOCK_MAP[keyword]
    if keyword.isdigit():
        return keyword
    return None


def build_news_keyword(user_input: str) -> str:
    code = resolve_stock_code(user_input)
    if code:
        return f"{code} 台股"
    return user_input.strip()


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_google_news_rss_url(keyword: str) -> str:
    q = urllib.parse.quote(keyword)
    return f"https://news.google.com/rss/search?q={q}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"


def fetch_news(keyword: str, limit: int = 3) -> List[Tuple[str, str, str]]:
    """
    回傳 [(title, link, pub_date), ...]
    """
    rss_url = get_google_news_rss_url(keyword)

    try:
        resp = requests.get(rss_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception as e:
        print("fetch_news error:", e)
        return []

    items = []
    for item in root.findall(".//item"):
        title = clean_text(item.findtext("title", default=""))
        link = clean_text(item.findtext("link", default=""))
        pub_date = clean_text(item.findtext("pubDate", default=""))

        if title and link:
            items.append((title, link, pub_date))

        if len(items) >= limit:
            break

    return items


def format_news_message(user_keyword: str, limit: int = 3) -> str:
    keyword = build_news_keyword(user_keyword)
    news_list = fetch_news(keyword, limit=limit)

    if not news_list:
        search_url = (
            "https://news.google.com/search?q="
            + urllib.parse.quote(keyword)
            + "&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        )
        return (
            f"目前暫時抓不到「{user_keyword}」的新聞。\n"
            "你可以先點這個搜尋連結：\n"
            f"{search_url}"
        )

    lines = [f"{user_keyword} 最新新聞："]
    for i, (title, link, _) in enumerate(news_list, start=1):
        lines.append(f"\n{i}. {title}")
        lines.append(link)

    return "\n".join(lines)


def build_analysis_text(keyword: str) -> str:
    code = resolve_stock_code(keyword)
    news_keyword = build_news_keyword(keyword)
    news_list = fetch_news(news_keyword, limit=3)

    lines = [f"你查詢的是：{keyword}"]

    if code:
        lines.append(f"辨識代碼：{code}")
    else:
        lines.append("目前找不到明確股票代碼，先提供相關新聞。")

    lines.append("")
    lines.append("近期新聞：")

    if news_list:
        for i, (title, link, _) in enumerate(news_list, start=1):
            lines.append(f"{i}. {title}")
            lines.append(link)
    else:
        search_url = (
            "https://news.google.com/search?q="
            + urllib.parse.quote(news_keyword)
            + "&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        )
        lines.append(search_url)

    lines.append("")
    lines.append("網站首頁：")
    lines.append(BASE_URL)

    return "\n".join(lines)


def build_help_text() -> str:
    return (
        "你可以這樣輸入：\n"
        "分析 2330\n"
        "分析 群創\n"
        "新聞 2330\n"
        "新聞 群創\n"
        "網址\n"
        "幫助\n\n"
        "網站首頁：\n"
        f"{BASE_URL}"
    )


def safe_reply_text(text: str, max_len: int = 4500) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len - 20] + "\n\n（內容過長，已截斷）"


# =========================
# 首頁
# =========================
@app.route("/", methods=["GET"])
def home():
    return "LINE Bot is running successfully."


# =========================
# Webhook
# =========================
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    except Exception as e:
        print("Webhook handle error:", e)
        abort(500)

    return "OK"


# =========================
# 訊息處理
# =========================
@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_text = event.message.text.strip()
    lower_text = user_text.lower()

    if user_text in ["網址", "連結", "網站", "首頁", "web", "url"]:
        reply_text = (
            "請點以下連結查看網站：\n"
            f"{BASE_URL}"
        )

    elif user_text.startswith("新聞"):
        keyword = user_text.replace("新聞", "", 1).strip()

        if not keyword:
            reply_text = (
                "請輸入要查詢的新聞，例如：\n"
                "新聞 2330\n"
                "新聞 群創"
            )
        else:
            reply_text = format_news_message(keyword, limit=3)

    elif user_text.startswith("分析"):
        keyword = user_text.replace("分析", "", 1).strip()

        if not keyword:
            reply_text = (
                "請輸入要分析的股票，例如：\n"
                "分析 2330\n"
                "分析 群創"
            )
        else:
            reply_text = build_analysis_text(keyword)

    elif lower_text in ["help", "幫助", "功能", "指令"]:
        reply_text = build_help_text()

    else:
        reply_text = build_help_text()

    reply_text = safe_reply_text(reply_text)

    try:
        with ApiClient(configuration) as api_client:
            messaging_api = MessagingApi(api_client)
            messaging_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=reply_text)]
                )
            )
    except Exception as e:
        print("Reply error:", e)


# =========================
# 啟動
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
