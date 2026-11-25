"""
GDELT Project(news_sentiment.fetch_news.GDELTNewsSource)を使って、過去の
ニュースセンチメントを遡及収集(バックフィル)するための一回実行型スクリプト。

RSSフィードは直近の記事しか取得できず過去に遡れないため、学習データとして
十分な期間のセンチメント時系列を貯めるには、GDELTのような過去を検索できる
ソースでの一括バックフィルが必要になる。本スクリプトは指定した期間を固定幅の
バケット(既定24時間)に分割し、バケットごとにGDELTへ問い合わせて
score_and_tag_articles/aggregate_currency_sentiment/derive_pair_sentiment
(run_sentiment_pipeline.pyの定期実行と同じ集計ロジック)を適用し、1バケット=
1行として sentiment_output/news_sentiment_{mode}.csv に追記する(そうしないと
全期間の記事が1行に潰れてしまい、時系列データとして使えなくなるため)。
記事単位データ(raw_articles_{mode}.csv)にも、既存の重複除外ロジックで蓄積する。

再実行時の安全性: 既にoutput-csvに存在するバケット(time列が一致)は
スキップする。長時間かかる処理が途中で中断しても、同じコマンドを再実行すれば
続きから進む(重複行が増えない)。

使い方:
    python scripts/backfill_gdelt_history.py --mode medium \\
        --start-date 2024-01-01 --end-date 2024-06-01 --bucket-hours 24

GDELT APIへの負荷・レート制限を考慮し、バケットごとに --sleep-seconds
(既定1秒)だけ待機してから次のリクエストを送る。
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd

from news_sentiment.fetch_news import GDELTNewsSource
from news_sentiment.sentiment_model import FinBertSentimentModel
from news_sentiment.aggregate import run_aggregation, append_to_csv, append_raw_articles

DEFAULT_SYMBOLS = ["GBPUSD", "EURUSD", "USDJPY", "AUDUSD", "GBPJPY", "EURJPY", "GOLD"]
OUTPUT_DIR = "sentiment_output"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "backfill_gdelt_history.log")


def setup_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
    console_handler = logging.StreamHandler()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[file_handler, console_handler],
    )


def load_processed_bucket_times(output_csv: str) -> set:
    """
    💡 再実行時の安全性のため、output_csvに既にある time 列の値を集合として
    読み込む。中断後の再実行で、既に処理済みのバケットを重複して追加しない。
    """
    if not os.path.exists(output_csv):
        return set()
    try:
        existing = pd.read_csv(output_csv, usecols=["time"])
        return set(pd.to_datetime(existing["time"], utc=True))
    except Exception as e:
        logging.warning(f"⚠️ {output_csv} の既存time列の読み込みに失敗しました(安全のため続行せず終了します): {e}")
        raise


def iter_buckets(start: datetime, end: datetime, bucket_hours: float):
    cur = start
    delta = timedelta(hours=bucket_hours)
    while cur < end:
        bucket_end = min(cur + delta, end)
        yield cur, bucket_end
        cur = bucket_end


def main():
    setup_logger()
    logging.info("backfill_gdelt_history.py ver1.0")

    parser = argparse.ArgumentParser(description="GDELTによる過去ニュースセンチメントのバックフィル")
    parser.add_argument("--mode", type=str, default="medium", choices=["medium", "long"])
    parser.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--start-date", type=str, required=True,
                         help="バックフィル開始日時(例: 2024-01-01 または 2024-01-01T00:00:00)。UTC解釈")
    parser.add_argument("--end-date", type=str, required=True,
                         help="バックフィル終了日時(例: 2024-06-01)。UTC解釈")
    parser.add_argument("--bucket-hours", type=float, default=24.0,
                         help="何時間ごとに1行(=1集計)として区切るか(既定24時間=日次)。"
                              "GDELTの1リクエストあたりの上限件数に達する警告が頻発する場合は"
                              "小さくすることを検討する")
    parser.add_argument("--sleep-seconds", type=float, default=1.0,
                         help="各リクエスト後の待機時間(GDELT APIへの配慮)")
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--raw-articles-csv", type=str, default=None)
    parser.add_argument("--backup-dir", type=str, default=None,
                         help="出力ファイルの複製先ディレクトリ(run_sentiment_pipeline.pyと同様)")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    output_csv = args.output_csv or os.path.join(OUTPUT_DIR, f"news_sentiment_{args.mode}.csv")
    raw_articles_csv = args.raw_articles_csv or os.path.join(OUTPUT_DIR, f"raw_articles_{args.mode}.csv")
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(raw_articles_csv) or ".", exist_ok=True)
    if args.backup_dir:
        os.makedirs(args.backup_dir, exist_ok=True)

    start = pd.to_datetime(args.start_date, utc=True).to_pydatetime()
    end = pd.to_datetime(args.end_date, utc=True).to_pydatetime()
    if start >= end:
        logging.error(f"❌ --start-date({start})が--end-date({end})以降になっています。")
        return

    buckets = list(iter_buckets(start, end, args.bucket_hours))
    processed = load_processed_bucket_times(output_csv)
    logging.info(
        f"📅 バックフィル対象期間: {start} 〜 {end}({len(buckets)}バケット、"
        f"{args.bucket_hours}時間刻み。うち既に処理済み: "
        f"{sum(1 for b, _ in buckets if pd.Timestamp(b, tz='UTC') in processed)}件)"
    )

    logging.info("🧠 FinBERTモデルをロード中...")
    model = FinBertSentimentModel()
    source = GDELTNewsSource()

    for i, (bucket_start, bucket_end) in enumerate(buckets, start=1):
        if pd.Timestamp(bucket_start, tz="UTC") in processed:
            logging.info(f"[{i}/{len(buckets)}] {bucket_start}〜{bucket_end} は処理済みのためスキップ")
            continue

        logging.info(f"[{i}/{len(buckets)}] {bucket_start}〜{bucket_end} を取得中...")
        articles = source.fetch_range(bucket_start, bucket_end)

        result_df, scored_df = run_aggregation(articles, model, symbols, timestamp=bucket_start)
        append_to_csv(result_df, output_csv)
        append_raw_articles(scored_df, raw_articles_csv)

        if args.backup_dir:
            append_to_csv(result_df, os.path.join(args.backup_dir, os.path.basename(output_csv)))
            append_raw_articles(scored_df, os.path.join(args.backup_dir, os.path.basename(raw_articles_csv)))

        time.sleep(args.sleep_seconds)

    logging.info("✨ バックフィル完了")


if __name__ == "__main__":
    main()
