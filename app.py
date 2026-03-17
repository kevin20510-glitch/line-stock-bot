import os
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional

import pandas as pd
import requests
import twstock
from flask import Flask, request, abort
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator

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


LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
PORT = int(os.getenv("PORT", "10000"))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

PRICE_CACHE_TTL = 300
NEWS_CACHE_TTL = 600


BULLISH_KEYWORDS = [
    "營收成長", "營收創高", "獲利成長", "eps", "擴產", "接單", "訂單", "法說會樂觀",
    "股利", "配息", "ai", "合作", "併購", "上調目標價", "調升評等", "利多",
    "簽約", "新品", "突破", "創高", "回購",
]

BEARISH_KEYWORDS = [
    "營收衰退", "虧損", "下修", "減產", "裁員", "違約", "訴訟", "調降評等", "降評",
    "事故", "停工", "停牌", "利空", "衰退", "風險", "重挫", "賠償", "詐欺", "內線",
    "破底",
]

SECTOR_KEYWORDS = {
    "AI / 半導體": ["台積", "晶片", "半導體", "晶圓", "封測", "ai", "gpu", "server", "chip"],
    "電子代工 / 硬體": ["代工", "伺服器", "筆電", "主機板", "oem", "odm", "pc"],
    "金融": ["銀行", "金控", "保險", "證券", "金融"],
    "航運 / 物流": ["航運", "貨櫃", "散裝", "物流", "運價"],
    "傳產 / 工業": ["工具機", "鋼鐵", "水泥", "塑化", "工業", "設備", "機械"],
    "生技 / 醫療": ["生技", "新藥", "醫療", "藥證", "臨床"],
    "能源 / 綠能": ["太陽能", "綠能", "風電", "儲能", "能源"],
}

MANUAL_NAME_TO_CODE = {
    "台積": "2330",
    "台積電": "2330",
    "群創": "3481",
    "群創光電": "3481",
    "聯電": "2303",
    "鴻海": "2317",
    "仁寶": "2324",
    "金寶": "2312",
    "東台": "4526",
    "瑞智": "4532",
    "國泰金": "2882",
    "富邦金": "2881",
    "元大台灣50": "0050",
}


@dataclass
class NewsItem:
    title: str
    published: str
    source: str
    link: str
    sentiment: str
    score: int
    matched_keywords: List[str]
    summary: str


def build_code_name_maps() -> Tuple[Dict[str, str], Dict[str, str]]:
    code_to_name: Dict[str, str] = {}
    name_to_code: Dict[str, str] = {}

    for code, info in twstock.codes.items():
        if not code.isdigit():
            continue
        name = getattr(info, "name", "") or ""
        if not name:
            continue
        code_to_name[code] = name
        name_to_code[name] = code

    for name, code in MANUAL_NAME_TO_CODE.items():
        name_to_code[name] = code

    return code_to_name, name_to_code


CODE_TO_NAME, NAME_TO_CODE = build_code_name_maps()


class StockAnalyzer:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.price_cache: Dict[str, Tuple[float, pd.DataFrame, str, str]] = {}
        self.news_cache: Dict[str, Tuple[float, List[NewsItem]]] = {}

    def normalize_symbol(self, symbol: str) -> str:
        s = symbol.strip().replace("　", " ")
        s = " ".join(s.split())

        prefixes = ["分析", "查詢", "股票", "股價"]
        for prefix in prefixes:
            if s.startswith(prefix):
                s = s[len(prefix):].strip()
                break

        return s

    def resolve_symbol(self, user_input: str) -> Tuple[str, str]:
        s = self.normalize_symbol(user_input)
        if not s:
            raise ValueError("請輸入台股代碼或中文名稱")

        if s in NAME_TO_CODE:
            code = NAME_TO_CODE[s]
            return code, CODE_TO_NAME.get(code, s)

        if s.isdigit() and len(s) == 4:
            if s in CODE_TO_NAME:
                return s, CODE_TO_NAME[s]
            raise ValueError(f"找不到台股代碼：{s}")

        if s in CODE_TO_NAME:
            return s, CODE_TO_NAME[s]

        lowered = s.lower()
        for name, code in NAME_TO_CODE.items():
            if lowered == name.lower():
                return code, CODE_TO_NAME.get(code, name)

        for code, name in CODE_TO_NAME.items():
            if s in name:
                return code, name

        raise ValueError(f"找不到『{user_input}』對應的台股代碼")

    def _get_price_cache(self, key: str):
        item = self.price_cache.get(key)
        if not item:
            return None
        ts, df, name, code = item
        if time.time() - ts <= PRICE_CACHE_TTL:
            print(f"價格快取命中: {key}")
            return df.copy(), name, code
        self.price_cache.pop(key, None)
        return None

    def _set_price_cache(self, key: str, df: pd.DataFrame, name: str, code: str):
        self.price_cache[key] = (time.time(), df.copy(), name, code)

    def _fetch_monthly_history(self, stock: twstock.Stock, year: int, month: int):
        try:
            return stock.fetch(year, month)
        except Exception as e:
            print(f"抓取 {stock.sid} {year}-{month:02d} 失敗: {e}")
            return []

    def get_price_history(self, symbol: str, months: int = 14) -> Tuple[pd.DataFrame, str, str]:
        code, name = self.resolve_symbol(symbol)
        cache_key = f"{code}|{months}"

        cached = self._get_price_cache(cache_key)
        if cached:
            return cached

        stock = twstock.Stock(code)
        today = datetime.now()
        rows = []

        for i in range(months):
            y = today.year
            m = today.month - i
            while m <= 0:
                y -= 1
                m += 12

            monthly = self._fetch_monthly_history(stock, y, m)
            if monthly:
                rows.extend(monthly)
            time.sleep(0.2)

        if not rows:
            raise ValueError(f"抓不到 {name}（{code}）的價格資料")

        data = []
        seen_dates = set()

        for r in rows:
            if r.date in seen_dates:
                continue
            seen_dates.add(r.date)

            if r.close is None:
                continue

            data.append({
                "Date": pd.to_datetime(r.date),
                "Open": float(r.open) if r.open is not None else None,
                "High": float(r.high) if r.high is not None else None,
                "Low": float(r.low) if r.low is not None else None,
                "Close": float(r.close) if r.close is not None else None,
                "Volume": float(r.capacity) if r.capacity is not None else 0.0,
            })

        df = pd.DataFrame(data)
        if df.empty:
            raise ValueError(f"抓不到 {name}（{code}）的價格資料")

        df = df.sort_values("Date").dropna(subset=["Close"]).set_index("Date")

        if len(df) < 30:
            raise ValueError(f"{name}（{code}）歷史資料不足，暫時無法分析")

        self._set_price_cache(cache_key, df, name, code)
        return df, name, code

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["MA20"] = SMAIndicator(out["Close"], window=20).sma_indicator()
        out["MA60"] = SMAIndicator(out["Close"], window=60).sma_indicator()
        out["RSI14"] = RSIIndicator(out["Close"], window=14).rsi()

        macd = MACD(out["Close"], window_fast=12, window_slow=26, window_sign=9)
        out["MACD_HIST"] = macd.macd_diff()

        out["AVG_VOL20"] = out["Volume"].rolling(20, min_periods=1).mean()
        out["RET20"] = out["Close"].pct_change(20) * 100
        out["RET60"] = out["Close"].pct_change(60) * 100
        out["HIGH_52W"] = out["Close"].rolling(252, min_periods=1).max()
        out["LOW_52W"] = out["Close"].rolling(252, min_periods=1).min()
        return out

    def classify_sector(self, code: str, name: str, news_titles: List[str]) -> str:
        text = " ".join([code, name] + news_titles).lower()
        scores = {}
        for sector, keywords in SECTOR_KEYWORDS.items():
            scores[sector] = sum(1 for kw in keywords if kw.lower() in text)
        best = max(scores.items(), key=lambda x: x[1])
        return best[0] if best[1] > 0 else "其他"

    def build_news_query(self, code: str, name: str) -> str:
        keywords = [code, name]
        return " OR ".join(f'"{k}"' for k in keywords if k)

    def summarize_title(self, title: str, sentiment: str, matched: List[str]) -> str:
        tone = {"利多": "偏正面", "利空": "偏負面", "中性": "中性"}[sentiment]
        if matched:
            key_text = "、".join(matched[:3])
            return f"{tone}：標題提到 {key_text}。"
        return f"{tone}：目前只從標題判讀，建議點進原文再確認細節。"

    def analyze_news_sentiment(self, text: str) -> Tuple[str, int, List[str], str]:
        lowered = text.lower()
        bull_matches = [kw for kw in BULLISH_KEYWORDS if kw.lower() in lowered]
        bear_matches = [kw for kw in BEARISH_KEYWORDS if kw.lower() in lowered]

        bull_score = len(bull_matches) * 2
        bear_score = len(bear_matches) * 2
        net = bull_score - bear_score

        if net >= 2:
            sentiment = "利多"
            score = min(10, net)
            matched = bull_matches
        elif net <= -2:
            sentiment = "利空"
            score = max(-10, net)
            matched = bear_matches
        else:
            sentiment = "中性"
            score = 0
            matched = bull_matches + bear_matches

        summary = self.summarize_title(text, sentiment, matched)
        return sentiment, score, matched, summary

    def resolve_google_news_link(self, url: str) -> str:
        if not url:
            return url

        try:
            resp = self.session.get(url, timeout=10, allow_redirects=True, stream=True)
            final_url = resp.url
            resp.close()
            return final_url or url
        except Exception as e:
            print("解析 Google News 連結失敗:", e)
            return url

    def shorten_url(self, url: str, max_len: int = 100) -> str:
        if not url:
            return url
        return url if len(url) <= max_len else url[:max_len] + "..."

    def fetch_google_news(self, code: str, name: str, days: int = 30, max_items: int = 5) -> List[NewsItem]:
        cache_key = f"{code}|{name}|{days}|{max_items}"
        item = self.news_cache.get(cache_key)
        if item:
            ts, cached_news = item
            if time.time() - ts <= NEWS_CACHE_TTL:
                print(f"新聞快取命中: {cache_key}")
                return cached_news

        query = self.build_news_query(code, name)
        rss_url = (
            "https://news.google.com/rss/search?hl=zh-TW&gl=TW&ceid=TW:zh-Hant&q="
            + urllib.parse.quote(query)
        )

        print("Google News RSS:", rss_url)

        resp = self.session.get(rss_url, timeout=20)
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        items = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        for item in root.findall("./channel/item"):
            title = (item.findtext("title") or "").strip()
            pub_date = (item.findtext("pubDate") or "").strip()

            raw_link = (item.findtext("link") or "").strip()
            link = self.resolve_google_news_link(raw_link)

            source_el = item.find("source")
            source = source_el.text.strip() if source_el is not None and source_el.text else "未知來源"

            try:
                published_dt = datetime.strptime(
                    pub_date, "%a, %d %b %Y %H:%M:%S GMT"
                ).replace(tzinfo=timezone.utc)
            except Exception:
                published_dt = datetime.now(timezone.utc)

            if published_dt < cutoff:
                continue

            sentiment, score, matched, summary = self.analyze_news_sentiment(title)

            items.append(
                NewsItem(
                    title=title,
                    published=published_dt.astimezone().strftime("%Y-%m-%d"),
                    source=source,
                    link=link,
                    sentiment=sentiment,
                    score=score,
                    matched_keywords=matched,
                    summary=summary,
                )
            )

            if len(items) >= max_items:
                break

        self.news_cache[cache_key] = (time.time(), items)
        return items

    def score_trend(self, row: pd.Series) -> Tuple[int, str]:
        score = 50

        close = float(row["Close"])
        ma20 = float(row["MA20"]) if pd.notna(row["MA20"]) else close
        ma60 = float(row["MA60"]) if pd.notna(row["MA60"]) else close
        rsi = float(row["RSI14"]) if pd.notna(row["RSI14"]) else 50
        macd_hist = float(row["MACD_HIST"]) if pd.notna(row["MACD_HIST"]) else 0
        ret20 = float(row["RET20"]) if pd.notna(row["RET20"]) else 0
        ret60 = float(row["RET60"]) if pd.notna(row["RET60"]) else 0
        avg_vol20 = float(row["AVG_VOL20"]) if pd.notna(row["AVG_VOL20"]) else 0
        vol = float(row["Volume"])
        high_52w = float(row["HIGH_52W"])
        low_52w = float(row["LOW_52W"])

        score += 8 if close > ma20 else -8
        score += 10 if close > ma60 else -10
        score += 8 if ma20 > ma60 else -8

        if 45 <= rsi <= 65:
            score += 6
        elif rsi > 75:
            score -= 6
        elif rsi < 30:
            score += 2
        else:
            score -= 2

        score += 6 if macd_hist > 0 else -6

        if ret20 > 8:
            score += 6
        elif ret20 < -8:
            score -= 6

        if ret60 > 15:
            score += 6
        elif ret60 < -15:
            score -= 6

        if avg_vol20 > 0:
            vol_ratio = vol / avg_vol20
            if 1.0 <= vol_ratio <= 2.5:
                score += 4
            elif vol_ratio > 4:
                score -= 4

        high_gap = (close / high_52w - 1) * 100 if high_52w else 0
        low_gap = (close / low_52w - 1) * 100 if low_52w else 0

        if high_gap > -10:
            score += 4
        if low_gap < 5:
            score -= 4

        score = max(0, min(100, round(score)))

        if score >= 75:
            signal = "強勢"
        elif score >= 60:
            signal = "偏多"
        elif score >= 45:
            signal = "中性"
        elif score >= 30:
            signal = "偏弱"
        else:
            signal = "弱勢"

        return score, signal

    def score_news(self, news: List[NewsItem]) -> Tuple[int, List[str], List[str], List[str]]:
        if not news:
            return 50, [], [], []

        total = sum(item.score for item in news)
        score = max(0, min(100, 50 + total * 4))

        good = [
            f"{n.published}｜{n.title}\n網址：{self.shorten_url(n.link)}"
            for n in news if n.sentiment == "利多"
        ]
        bad = [
            f"{n.published}｜{n.title}\n網址：{self.shorten_url(n.link)}"
            for n in news if n.sentiment == "利空"
        ]
        brief = [
            f"{n.published}｜{n.source}｜{n.summary}\n網址：{self.shorten_url(n.link)}"
            for n in news[:2]
        ]

        return round(score), good[:2], bad[:2], brief

    def classify_stock(self, total_score: int, news_score: int, row: pd.Series) -> str:
        close = float(row["Close"])
        ma20 = float(row["MA20"]) if pd.notna(row["MA20"]) else close
        ma60 = float(row["MA60"]) if pd.notna(row["MA60"]) else close
        rsi = float(row["RSI14"]) if pd.notna(row["RSI14"]) else 50
        ret20 = float(row["RET20"]) if pd.notna(row["RET20"]) else 0

        if total_score >= 78 and close > ma20 > ma60 and news_score >= 55:
            return "成長強勢股"
        if total_score >= 65 and rsi < 70 and ret20 > 0:
            return "趨勢觀察股"
        if 45 <= total_score < 65:
            return "區間整理股"
        if total_score < 45 and news_score < 45:
            return "消息偏空股"
        return "高波動觀察股"

    def suggest_action(self, total_score: int, trend_score: int, news_score: int, row: pd.Series) -> str:
        close = float(row["Close"])
        ma20 = float(row["MA20"]) if pd.notna(row["MA20"]) else close
        ma60 = float(row["MA60"]) if pd.notna(row["MA60"]) else close
        rsi = float(row["RSI14"]) if pd.notna(row["RSI14"]) else 50

        if total_score >= 75 and trend_score >= 70 and news_score >= 55 and close > ma20 > ma60 and rsi < 75:
            return "買進"
        if total_score >= 50 and close >= ma60:
            return "觀望"
        return "避開"

    def analyze_stock_text(self, user_input: str) -> str:
        query = self.normalize_symbol(user_input)
        print("analyze_stock_text query:", query)

        df, name, code = self.get_price_history(query)
        print("價格資料筆數:", len(df))

        df = self.compute_indicators(df)
        row = df.iloc[-1]
        news = self.fetch_google_news(code, name)

        trend_score, signal = self.score_trend(row)
        news_score, good_news, bad_news, news_brief = self.score_news(news)
        total_score = round(trend_score * 0.65 + news_score * 0.35)
        sector_guess = self.classify_sector(code, name, [n.title for n in news])
        category = self.classify_stock(total_score, news_score, row)
        action = self.suggest_action(total_score, trend_score, news_score, row)

        current_price = float(row["Close"])
        ret20 = float(row["RET20"]) if pd.notna(row["RET20"]) else 0
        ret60 = float(row["RET60"]) if pd.notna(row["RET60"]) else 0

        lines = [
            f"{name}（{code}）",
            f"現價：{current_price:.2f}",
            f"20日漲跌：{ret20:.2f}%｜60日漲跌：{ret60:.2f}%",
            f"總分：{total_score}/100｜技術：{trend_score}｜新聞：{news_score}",
            f"趨勢：{signal}｜分類：{category}",
            f"題材：{sector_guess}｜建議：{action}",
        ]

        if good_news:
            lines.append("利多新聞：\n" + good_news[0])

        if bad_news:
            lines.append("利空新聞：\n" + bad_news[0])

        if news_brief:
            lines.append("新聞重點：\n" + news_brief[0])

        reply = "\n".join(lines)

        if len(reply) > 4500:
            reply = reply[:4500] + "\n（內容過長，已自動截斷）"

        return reply


analyzer = StockAnalyzer()
app = Flask(__name__)
handler = WebhookHandler(LINE_CHANNEL_SECRET)


def get_help_text() -> str:
    return (
        "請直接輸入台股代碼或中文名稱。\n"
        "例如：2330、台積電、3481、群創\n\n"
        "可用指令：\n"
        "help：顯示說明\n"
        "分析台積電\n"
        "分析 2330\n"
        "查詢群創"
    )


@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    print("收到 webhook body:", body[:500])

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Signature 驗證失敗")
        abort(400)
    except Exception as e:
        print("Webhook error:", e)
        abort(500)

    return "OK"


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_text = event.message.text.strip()

    if user_text.lower() in ["help", "說明", "幫助"]:
        reply_text = get_help_text()
    else:
        try:
            query = analyzer.normalize_symbol(user_text)
            print("收到訊息:", repr(user_text))
            print("整理後查詢:", repr(query))
            code, name = analyzer.resolve_symbol(query)
            print("解析代碼:", code, name)
            reply_text = analyzer.analyze_stock_text(query)
        except Exception as e:
            print("分析失敗:", e)
            reply_text = f"分析失敗：{e}\n\n{get_help_text()}"

    configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_text)]
            )
        )


@app.route("/", methods=["GET"])
def home():
    return "LINE 台股機器人運作中"


if __name__ == "__main__":
    if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
        raise ValueError("請先設定 LINE_CHANNEL_ACCESS_TOKEN 與 LINE_CHANNEL_SECRET 環境變數")
    app.run(host="0.0.0.0", port=PORT)
