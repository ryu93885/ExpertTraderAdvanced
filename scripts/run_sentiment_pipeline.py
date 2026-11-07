"""
ニュースセンチメント・パイプラインの実行エントリーポイント。

cron(VPS)やタスクスケジューラ(ローカルPC)から定期的に(例: 1時間ごと)起動する
ことを想定した一回実行型のスクリプト。独自のループ・スリープ処理は持たせず、
外部のスケジューラに実行自体を任せる設計にしている(ExpertTrader側の
portfolio_trading_bot.pyのような常駐プロセス方式ではなく、cron向けの定期バッチ)。

実行例(どのディレクトリから実行しても動く。python -m scripts.run_sentiment_pipeline
でも実行できるが、その場合はリポジトリルートをカレントディレクトリにする必要がある):
    python scripts/run_sentiment_pipeline.py --mode medium --lookback-hours 6
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler

# 💡 修正(2026-11-06): news_sentiment.xxx という絶対import自体は正しいが、
# 「python -m scripts.run_sentiment_pipeline」はカレントディレクトリが
# リポジトリルートと完全に一致している場合しかnews_sentimentパッケージを
# 解決できず(scriptsパッケージ自体は見つかるのに news_sentiment の方だけ
# 見つからない、という分かりにくい失敗の仕方をする)、実行場所に依存する
# 脆さがあった。このファイル自身の位置(__file__)からリポジトリルートを
# 逆算してsys.pathに加えることで、どのディレクトリから・どちらの実行方法
# (直接実行 / -m実行)でも確実にnews_sentimentパッケージを解決できるようにする。
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from news_sentiment.fetch_news import RSSNewsSource
from news_sentiment.sentiment_model import FinBertSentimentModel
from news_sentiment.aggregate import run_aggregation, append_to_csv

DEFAULT_SYMBOLS = ["GBPUSD", "EURUSD", "USDJPY", "AUDUSD", "GBPJPY", "EURJPY", "GOLD"]
OUTPUT_DIR = "sentiment_output"


LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "run_sentiment_pipeline.log")


def setup_logger():
    # 💡 追加(2026-11-07): cron/タスクスケジューラからの定期実行では、コンソール出力
    # (標準出力)は呼び出し元が明示的にリダイレクトしない限りどこにも残らない。
    # 1時間ごと等で長期間動かし続ける運用を想定し、スクリプト側で確実にファイルへ
    # 記録するようにする。1ファイルに全実行分を追記していくと際限なく肥大化するため、
    # サイズベースでローテーションする(5MB×5世代を上限とする)。
    os.makedirs(LOG_DIR, exist_ok=True)
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
    console_handler = logging.StreamHandler()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[file_handler, console_handler],
    )


def main():
    setup_logger()
    logging.info("run_sentiment_pipeline.py ver1.1")

    parser = argparse.ArgumentParser(description="ニュースセンチメントの定期集計パイプライン")
    # 💡 条件6: ExpertTraderと異なり、short(スキャルピング帯)は対象外。
    parser.add_argument("--mode", type=str, default="medium", choices=["medium", "long"],
                         help="対象モード。medium/longのみ対応(shortは対象外)")
    parser.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS),
                         help="対象銘柄(カンマ区切り)")
    parser.add_argument("--lookback-hours", type=float, default=6.0,
                         help="何時間前までの記事を対象にするか")
    parser.add_argument("--output-csv", type=str, default=None,
                         help="出力CSVのパス。省略時は sentiment_output/news_sentiment_{mode}.csv")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    output_csv = args.output_csv or os.path.join(OUTPUT_DIR, f"news_sentiment_{args.mode}.csv")
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    since = datetime.now(timezone.utc) - timedelta(hours=args.lookback_hours)

    logging.info(f"📡 ニュースを収集中(過去{args.lookback_hours}時間分)...")
    source = RSSNewsSource()
    articles = source.fetch_recent(since)

    if not articles:
        logging.warning("⚠️ 対象期間内の記事が0件でした。センチメントはすべて0(ニュートラル)として記録します。")

    logging.info("🧠 FinBERTモデルをロード中...")
    model = FinBertSentimentModel()

    logging.info("📊 センチメントを集計中...")
    result_df = run_aggregation(articles, model, symbols)

    append_to_csv(result_df, output_csv)
    logging.info("✨ パイプライン完了")


if __name__ == "__main__":
    main()
