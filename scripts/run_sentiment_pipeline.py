"""
ニュースセンチメント・パイプラインの実行エントリーポイント。

cron(VPS)やタスクスケジューラ(ローカルPC)から定期的に(例: 1時間ごと)起動する
ことを想定した一回実行型のスクリプト。独自のループ・スリープ処理は持たせず、
外部のスケジューラに実行自体を任せる設計にしている(ExpertTrader側の
portfolio_trading_bot.pyのような常駐プロセス方式ではなく、cron向けの定期バッチ)。

実行例(どのディレクトリから実行しても動く。python -m scripts.run_sentiment_pipeline
でも実行できるが、その場合はリポジトリルートをカレントディレクトリにする必要がある):
    python scripts/run_sentiment_pipeline.py --mode medium --lookback-hours 6

rclone等でGoogle Driveへ同期しているローカルフォルダがあれば、--backup-dirで
指定すると、出力ファイル(集計結果CSV・記事単位CSV)をそこにも複製する:
    python scripts/run_sentiment_pipeline.py --mode medium --lookback-hours 6 \
        --backup-dir "C:\\path\\to\\rclone-synced-folder"
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

from news_sentiment.fetch_news import RSSNewsSource, GDELTNewsSource
from news_sentiment.sentiment_model import FinBertSentimentModel
from news_sentiment.aggregate import run_aggregation, append_to_csv, append_raw_articles

DEFAULT_SYMBOLS = ["GBPUSD", "EURUSD", "USDJPY", "AUDUSD", "GBPJPY", "EURJPY", "GOLD"]
OUTPUT_DIR = "sentiment_output"

# 💡 追加(2026-11-25): RSS(低遅延だが過去に遡れない)とGDELT(過去に遡れるが
# 処理に遅延があり個別の小規模ソースを必ず拾える保証はない)は、互いの弱点を補う
# 関係にあるため、どちらも/両方を選べるようにする。NewsSourceの共通インターフェース
# のおかげで、articlesリストを単純に連結するだけで両方の結果を同じパイプライン
# (currency_tags.py・FinBERT・aggregate.py)にそのまま流せる。
SOURCE_FACTORIES = {
    "rss": RSSNewsSource,
    "gdelt": GDELTNewsSource,
}


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
    logging.info("run_sentiment_pipeline.py ver1.3")

    parser = argparse.ArgumentParser(description="ニュースセンチメントの定期集計パイプライン")
    # 💡 条件6: ExpertTraderと異なり、short(スキャルピング帯)は対象外。
    parser.add_argument("--mode", type=str, default="medium", choices=["medium", "long"],
                         help="対象モード。medium/longのみ対応(shortは対象外)")
    parser.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS),
                         help="対象銘柄(カンマ区切り)")
    parser.add_argument("--lookback-hours", type=float, default=6.0,
                         help="何時間前までの記事を対象にするか")
    parser.add_argument("--output-csv", type=str, default=None,
                         help="集計結果CSVのパス。省略時は sentiment_output/news_sentiment_{mode}.csv")
    parser.add_argument("--raw-articles-csv", type=str, default=None,
                         help="記事単位データCSVのパス。省略時は sentiment_output/raw_articles_{mode}.csv")
    # 💡 追加(2026-11-25): RSSとGDELTを併用できるようにする。デフォルトはrssのみ
    # (従来通りの挙動)。両方使う場合は --sources rss,gdelt のように指定する。
    parser.add_argument("--sources", type=str, default="rss",
                         help=f"使用するニュース取得元(カンマ区切り、{','.join(SOURCE_FACTORIES)}から選択)。"
                              "省略時はrssのみ(従来通り)")
    # 💡 追加(2026-11-10): ローカルPC/VPS運用では、手動でのファイル移動・同期作業を
    # 完全には無くせない(ローカル・GitHub・Google Drive間を行き来する構成のため)。
    # 既存のrclone等によるDrive同期フォルダを指定すれば、出力ファイルをそこにも
    # 書き込む(プログラム内で2か所に保存する)ことで、記事データの消失を防ぐ。
    # 省略時は従来通り1か所のみへの保存(この機能が無い環境でも影響しない)。
    parser.add_argument("--backup-dir", type=str, default=None,
                         help="出力ファイルの複製先ディレクトリ(例: rcloneがDriveへ同期して"
                              "いるローカルフォルダ)。省略時はバックアップしない")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    output_csv = args.output_csv or os.path.join(OUTPUT_DIR, f"news_sentiment_{args.mode}.csv")
    raw_articles_csv = args.raw_articles_csv or os.path.join(OUTPUT_DIR, f"raw_articles_{args.mode}.csv")
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(raw_articles_csv) or ".", exist_ok=True)
    if args.backup_dir:
        os.makedirs(args.backup_dir, exist_ok=True)

    since = datetime.now(timezone.utc) - timedelta(hours=args.lookback_hours)

    source_names = [s.strip().lower() for s in args.sources.split(",") if s.strip()]
    unknown = [s for s in source_names if s not in SOURCE_FACTORIES]
    if unknown:
        logging.error(f"❌ 未知のニュース取得元が指定されました: {unknown}(有効な値: {list(SOURCE_FACTORIES)})")
        return

    logging.info(f"📡 ニュースを収集中(過去{args.lookback_hours}時間分、取得元={source_names})...")
    articles = []
    for name in source_names:
        src_articles = SOURCE_FACTORIES[name]().fetch_recent(since)
        logging.info(f"  [{name}] {len(src_articles)}件")
        articles.extend(src_articles)

    if not articles:
        logging.warning("⚠️ 対象期間内の記事が0件でした。センチメントはすべて0(ニュートラル)として記録します。")

    logging.info("🧠 FinBERTモデルをロード中...")
    model = FinBertSentimentModel()

    logging.info("📊 センチメントを集計中...")
    result_df, scored_df = run_aggregation(articles, model, symbols)

    append_to_csv(result_df, output_csv)
    append_raw_articles(scored_df, raw_articles_csv)

    if args.backup_dir:
        logging.info(f"📥 バックアップ先にも保存します: {args.backup_dir}")
        append_to_csv(result_df, os.path.join(args.backup_dir, os.path.basename(output_csv)))
        append_raw_articles(scored_df, os.path.join(args.backup_dir, os.path.basename(raw_articles_csv)))

    logging.info("✨ パイプライン完了")


if __name__ == "__main__":
    main()
