"""
記事がどの通貨(またはGOLD)に関連するかを、キーワード一致で判定する。

設計意図: ExpertTrader側の7銘柄(GBPUSD, EURUSD, USDJPY, AUDUSD, GBPJPY, EURJPY,
GOLD)はいずれも2通貨(またはGOLD)の組み合わせであり、ニュースの影響も通貨単位で
考えるのが自然(例:「米利上げ」はUSDJPYにもEURUSDにも影響するが、向きが逆)。
このモジュールでは通貨単位でタグ付けし、ペア単位のセンチメントへの変換は
aggregate.py側(例: EURUSD_sentiment = EUR_sentiment - USD_sentiment)で行う。
"""

import re
from typing import Dict, List

from news_sentiment.fetch_news import NewsArticle

# 💡 キーワードは小文字で比較する(マッチング時に.lower()する)。
# 誤検出を抑えるため、"AUD" のような3文字の通貨コード単体は誤爆しやすい
# (他の単語の一部に含まれる等)ため、識別力の高い語(国名・中銀名・通貨名の
# フルスペル等)を中心に構成している。
CURRENCY_KEYWORDS: Dict[str, List[str]] = {
    # 💡 修正(2026-11-10の実機データで判明した誤検出への対応): 汎用的な"dollar"
    # (国名の修飾語なし)をUSDのキーワードに含めていたため、"New Zealand Dollar"
    # 等、他国の「~ドル」を指す記事がUSDとしても誤タグ付けされていた(実例:
    # NZD/RBNZに関する記事がUSDとしても集計されてしまっていた)。"us dollar"/
    # "u.s. dollar"/"usd"等の、USDであることが明確な語のみを残す。
    "USD": ["usd", "u.s. dollar", "us dollar", "federal reserve", "the fed",
            "fomc", "united states", "u.s. economy", "non-farm payroll", "nonfarm payroll"],
    "EUR": ["eur", "euro", "eurozone", "european central bank", "ecb", "european union"],
    "GBP": ["gbp", "pound sterling", "british pound", "bank of england", "boe", "uk economy", "united kingdom"],
    "JPY": ["jpy", "japanese yen", "yen", "bank of japan", "boj", "japan economy"],
    "AUD": ["aud", "australian dollar", "reserve bank of australia", "rba", "australia economy"],
    # 💡 修正(2026-11-13の実機データ・追加検証で判明した誤検出への対応):
    # "safe haven"/"safe-haven"はJPY・USD・CHF等にも日常的に使われる汎用語で
    # あり、GOLDに一切言及していない記事でも誤タグ付けされていた(実例:
    # raw_articles_medium.csvの実データで、USD単体の記事が"safe-haven"により
    # GOLDにも誤って集計されていた)。"precious metal"も銀・白金等、GOLD以外の
    # 貴金属の記事にも使われるため同様に除外する。"gold"そのものにも
    # "gold standard"(比喩表現)等の誤検出リスクが残るが、XAU/bullionと比べ
    # どれだけ実害があるか実データでの追加確認が必要なため、今回はそのまま残す。
    "GOLD": ["gold", "xau", "bullion"],
}


def tag_currencies(article: NewsArticle) -> List[str]:
    """
    記事のタイトル+要約を走査し、CURRENCY_KEYWORDSに一致した通貨コードの
    リストを返す(複数一致あり得る)。どれにも一致しなければ空リスト。
    """
    text = article.full_text.lower()
    matched = []
    for currency, keywords in CURRENCY_KEYWORDS.items():
        if any(re.search(r"\b" + re.escape(kw) + r"\b", text) for kw in keywords):
            matched.append(currency)
    return matched
