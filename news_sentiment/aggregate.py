"""
記事の収集→通貨タグ付け→FinBERTセンチメント推論→通貨別・銘柄ペア別の集計を行う。

出力は、ExpertTrader側の precompute_portfolio.py が生成する "*_with_preds.csv"
と同じ発想(時刻をキーに持つ、後から結合しやすいCSV)にしている。将来的に
ExpertTraderと組み合わせる際は、時刻をキーにこのCSVをExpertTrader側のマージ済み
データ(portfolio_merged_data_*.csv)と突き合わせ、追加の観測特徴量として
PortfolioFXEnvの観測ベクトルに組み込むことを想定している(ただし、その統合自体は
本リポジトリのスコープ外。条件5の「将来的な組み合わせ」に備えた設計のみ行う)。
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

from fetch_news import NewsArticle
from currency_tags import tag_currencies
from sentiment_model import FinBertSentimentModel

logger = logging.getLogger(__name__)

# 💡 ExpertTrader側のTARGET_SYMBOLSと対応する、銘柄→(基軸通貨, 決済通貨)のマップ。
# 標準的なFXの建値規則通り、基軸通貨が強い(ポジティブ)ほどペアは上昇方向、
# 決済通貨が強いほど下落方向になる。GOLDは通貨ペアではないため、決済通貨側は
# Noneとし、GOLD自体のセンチメントバケットをそのまま使う。
SYMBOL_CURRENCY_PAIRS = {
    "GBPUSD": ("GBP", "USD"),
    "EURUSD": ("EUR", "USD"),
    "USDJPY": ("USD", "JPY"),
    "AUDUSD": ("AUD", "USD"),
    "GBPJPY": ("GBP", "JPY"),
    "EURJPY": ("EUR", "JPY"),
    "GOLD": ("GOLD", None),
}


def score_and_tag_articles(articles: List[NewsArticle], model: FinBertSentimentModel) -> pd.DataFrame:
    """
    記事リストを受け取り、1記事1行(複数通貨にタグ付けされた記事は複数行に展開)の
    DataFrameを返す。列: title, source, published, currency, compound_sentiment
    """
    if not articles:
        return pd.DataFrame(columns=["title", "source", "published", "currency", "compound_sentiment"])

    texts = [a.full_text for a in articles]
    scores = model.score_texts(texts)

    rows = []
    for article, prob_dict in zip(articles, scores):
        currencies = tag_currencies(article)
        if not currencies:
            continue  # どの通貨にも関連しない記事(一般ニュース等)は集計対象外
        compound = FinBertSentimentModel.to_compound_score(prob_dict)
        for currency in currencies:
            rows.append({
                "title": article.title,
                "source": article.source,
                "published": article.published,
                "currency": currency,
                "compound_sentiment": compound,
            })

    return pd.DataFrame(rows)


def aggregate_currency_sentiment(scored_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    通貨ごとに平均センチメントと記事件数を集計する。
    戻り値: {"USD": {"avg_sentiment": 0.12, "article_count": 5}, ...}
    """
    result = {}
    if scored_df.empty:
        return result
    grouped = scored_df.groupby("currency")["compound_sentiment"].agg(["mean", "count"])
    for currency, row in grouped.iterrows():
        result[currency] = {"avg_sentiment": float(row["mean"]), "article_count": int(row["count"])}
    return result


def derive_pair_sentiment(currency_sentiment: Dict[str, Dict[str, float]], symbols: List[str]) -> Dict[str, float]:
    """
    通貨単位のセンチメントから、銘柄(通貨ペア)ごとのセンチメント差分を導出する。
    記事が無く該当通貨のセンチメントが存在しない場合は0(ニュートラル)として扱う。
    """
    pair_sentiment = {}
    for symbol in symbols:
        if symbol not in SYMBOL_CURRENCY_PAIRS:
            logger.warning(f"⚠️ {symbol} の通貨ペア定義が見つかりません。スキップします。")
            continue
        base, quote = SYMBOL_CURRENCY_PAIRS[symbol]
        base_score = currency_sentiment.get(base, {}).get("avg_sentiment", 0.0)
        if quote is None:
            pair_sentiment[symbol] = base_score
        else:
            quote_score = currency_sentiment.get(quote, {}).get("avg_sentiment", 0.0)
            pair_sentiment[symbol] = base_score - quote_score
    return pair_sentiment


def run_aggregation(articles: List[NewsArticle], model: FinBertSentimentModel, symbols: List[str],
                     timestamp: Optional[datetime] = None) -> pd.DataFrame:
    """
    収集済みの記事リストから、1行(=1実行時刻)・銘柄数分の列を持つサマリーDataFrameを
    作る。ExpertTrader側のCSVと時刻でjoinしやすいよう、時刻は1列にまとめている。
    列: time, {symbol}_news_sentiment (銘柄の数だけ), {symbol}_news_article_count
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    scored_df = score_and_tag_articles(articles, model)
    currency_sentiment = aggregate_currency_sentiment(scored_df)
    pair_sentiment = derive_pair_sentiment(currency_sentiment, symbols)

    row = {"time": timestamp}
    for symbol in symbols:
        row[f"{symbol}_news_sentiment"] = pair_sentiment.get(symbol, 0.0)
        base, quote = SYMBOL_CURRENCY_PAIRS.get(symbol, (None, None))
        count = currency_sentiment.get(base, {}).get("article_count", 0)
        if quote is not None:
            count += currency_sentiment.get(quote, {}).get("article_count", 0)
        row[f"{symbol}_news_article_count"] = count

    logger.info(f"📊 センチメント集計結果: { {k: v for k, v in row.items() if k != 'time'} }")
    return pd.DataFrame([row])


def append_to_csv(df_row: pd.DataFrame, csv_path: str) -> None:
    """1実行分の集計結果を、既存CSVに追記する(無ければヘッダー付きで新規作成)。"""
    import os
    file_exists = os.path.exists(csv_path)
    df_row.to_csv(csv_path, mode="a", header=not file_exists, index=False)
    logger.info(f"💾 センチメント結果を保存しました: {csv_path}")
