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
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

from news_sentiment.fetch_news import NewsArticle
from news_sentiment.currency_tags import tag_currencies
from news_sentiment.sentiment_model import FinBertSentimentModel

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


RAW_ARTICLE_COLUMNS = ["title", "summary", "url", "source", "published", "currency", "compound_sentiment"]
# 💡 記事の一意性判定(重複除外)に使うキー列。--lookback-hoursを実行間隔より長めに
# 設定しているため、同じ記事が複数回の実行で再度ヒットする。urlは一部フィードで
# 欠落/不安定なことがあるため使わず、(source, title, published)の組で判定する。
RAW_ARTICLE_DEDUP_KEYS = ["source", "title", "published"]


def score_and_tag_articles(articles: List[NewsArticle], model: FinBertSentimentModel) -> pd.DataFrame:
    """
    記事リストを受け取り、1記事1行(複数通貨にタグ付けされた記事は複数行に展開)の
    DataFrameを返す。列: RAW_ARTICLE_COLUMNS参照。

    💡 追加(2026-11-10): summary・urlも保持するようにした。集計(平均センチメント)
    には使わないが、記事そのものの蓄積・バックアップ(将来FinBERTの差し替えや
    キーワード判定の改善時に、実際に推論へ入力したテキスト(title+summary)を
    使って過去の記事を再採点できるようにする)に必要なため。
    """
    if not articles:
        return pd.DataFrame(columns=RAW_ARTICLE_COLUMNS)

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
                "summary": article.summary,
                "url": article.url,
                "source": article.source,
                "published": article.published,
                "currency": currency,
                "compound_sentiment": compound,
            })

    return pd.DataFrame(rows, columns=RAW_ARTICLE_COLUMNS)


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
                     timestamp: Optional[datetime] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    収集済みの記事リストから、1行(=1実行時刻)・銘柄数分の列を持つサマリーDataFrameを
    作る。ExpertTrader側のCSVと時刻でjoinしやすいよう、時刻は1列にまとめている。
    列: time, {symbol}_news_sentiment (銘柄の数だけ), {symbol}_news_article_count

    💡 修正(2026-11-10): 戻り値を (summary_df, scored_df) のタプルに変更した。
    scored_df(記事単位のデータ)は集計に使った後に捨てていたが、これも呼び出し元で
    蓄積・バックアップできるようにするため、合わせて返す。
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
    return pd.DataFrame([row]), scored_df


def append_to_csv(df_row: pd.DataFrame, csv_path: str) -> None:
    """1実行分の集計結果を、既存CSVに追記する(無ければヘッダー付きで新規作成)。"""
    file_exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df_row.to_csv(csv_path, mode="a", header=not file_exists, index=False)
    logger.info(f"💾 センチメント結果を保存しました: {csv_path}")


def append_raw_articles(scored_df: pd.DataFrame, csv_path: str) -> int:
    """
    💡 追加(2026-11-10): 記事単位のデータ(title・summary・url・source・published・
    currency・compound_sentiment)を、重複を除いて蓄積する。RSSフィードは直近の
    記事しか取得できず、一度流れた記事は再取得できないため、将来の再採点・
    キーワード判定の改善・Model B(センチメント予測モデル)の学習データ作成に
    使えるよう、生データとして残しておく。

    --lookback-hoursを実行間隔より長めに設定しているため、同じ記事が複数回の
    実行で再度ヒットする。(source, title, published)をキーに、既存ファイルに
    まだ無い行だけを追記する(ファイル全体を毎回書き直すのではなく、新規分のみ
    追記するため、件数が増えても1回あたりの処理は軽い)。

    戻り値: 実際に追記した件数。
    """
    if scored_df.empty:
        return 0

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    def _build_keys(df: pd.DataFrame):
        # 💡 CSVに書き出すとpublished(datetime)は文字列化されるため、再読み込み時は
        # 文字列、メモリ上のscored_dfではdatetimeオブジェクトのままになる。型が違うと
        # タプル比較が常に不一致になり重複を検出できないため、pd.to_datetime(utc=True)で
        # 両方を同じ型(Timestamp)に正規化してから比較する。
        published_norm = pd.to_datetime(df["published"], utc=True)
        return list(zip(df["source"], df["title"], published_norm))

    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path, usecols=RAW_ARTICLE_DEDUP_KEYS)
        existing_keys = set(_build_keys(existing_df))
        new_keys = _build_keys(scored_df)
        new_mask = [key not in existing_keys for key in new_keys]
        new_rows_df = scored_df[new_mask]
    else:
        new_rows_df = scored_df

    if new_rows_df.empty:
        logger.info(f"📰 新規記事0件(重複除外後)のため、{csv_path} への追記はありません")
        return 0

    file_exists = os.path.exists(csv_path)
    new_rows_df.to_csv(csv_path, mode="a", header=not file_exists, index=False)
    logger.info(f"💾 記事データを{len(new_rows_df)}件追記しました: {csv_path}")
    return len(new_rows_df)
