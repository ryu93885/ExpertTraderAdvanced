# ExpertTraderAdvanced

ExpertTrader(価格・テクニカル指標ベースのポートフォリオAIトレードシステム)を補完する、
経済ニュース・地政学リスクの文章解析モジュール。medium/longモードの取引判断には
価格データだけでなく、ニュースが織り込む材料の考慮が必要という判断から、新規プロジェクトとして立ち上げた。

## 条件・制約(2026-11-06時点の合意事項)

1. ExpertTrader内のプログラムは必要に応じてコピーして使用可（`vendor/` 等に配置予定）。
2. 文章解析モデルは**転移学習のみ**で使用する(ゼロから事前学習しない)。ベースモデルは
   [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)(金融テキストで学習済みの
   BERTベースモデル)を採用し、本体は凍結して使う。将来的に精度を追い込む場合も、
   追加の分類ヘッドのみを転移学習する想定。
3. **ローカルPCまたはX-ServerなどのVPSで実行できること**(GPU前提にしない)。
   FinBERTはBERT-baseサイズ(約1.1億パラメータ)で、定期バッチ推論程度の頻度であれば
   CPU推論で現実的な時間内に処理できる。
4. 将来的にExpertTraderと組み合わせて使うことを想定し、出力はExpertTraderの
   `precompute_portfolio.py`が生成する`_with_preds.csv`と同じ思想(時刻・銘柄をキーに
   持つ、後から結合しやすいCSV)で設計する。
5. 対象モードはmedium・longのみ(shortはスキャルピング帯で保有時間が短く、経済ニュース
   の影響を受けにくいこと、成績面でも採用予定がないことから対象外)。

## アーキテクチャ

```
news_sentiment/
  sentiment_model.py   … FinBERTのロードと推論
  fetch_news.py         … ニュース記事の収集(NewsSource抽象クラス + RSS実装)
  currency_tags.py      … 記事を通貨(USD/EUR/GBP/JPY/AUD/GOLD)にタグ付け
  aggregate.py           … 通貨別センチメント集計 → 銘柄ペア別センチメントの導出・CSV出力
scripts/
  run_sentiment_pipeline.py … 上記を定期実行するエントリーポイント(cron/タスクスケジューラ想定)
```

### ニュース取得元: RSSフィード(将来の差し替えを前提とした設計)

ニュース収集は `NewsSource` という抽象クラスの背後に隠してある。現時点では無料で
使えるRSSフィード(`RSSNewsSource`、ForexLive/DailyFX/Investing.com/FXStreet等を
`fetch_news.py`の`RSS_FEEDS`に列挙)のみを実装しているが、将来的に商用ニュースAPI等へ
切り替えたくなった場合は、`NewsSource`を継承した新しいクラスを追加するだけでよく、
`aggregate.py`以降のパイプラインは変更不要。

⚠️ `RSS_FEEDS`に列挙したURLは、一般に公開されている主要な金融・FX系ニュースサイトの
フィードだが、本セッション(Claude Codeのサンドボックス環境)はネットワークポリシー上
これら外部サイトに直接接続できず、疎通確認ができていない。ローカルPC/VPSで実際に
実行する際に、一度動作確認(記事が0件でないか)をしてください。フィードが無効化されて
いた場合は、該当URLをリストから削除・差し替えるだけで済みます。

### 通貨ペア vs 通貨単位での集計について

ExpertTraderの対象7銘柄(GBPUSD, EURUSD, USDJPY, AUDUSD, GBPJPY, EURJPY, GOLD)は、
いずれも2つの通貨(またはGOLD)の組み合わせです。「USD利上げ」のようなニュースは
USDJPY・EURUSD・AUDUSD・GBPUSD全てに同時に影響しますが、影響の向きはペアによって
異なります(USD高ニュース → USDJPY上昇要因だがEURUSD下落要因)。

そのため、ニュースのセンチメントは**通貨単位(USD/EUR/GBP/JPY/AUD + GOLD/地政学)**で
集計し、各通貨ペアのセンチメント特徴量は「構成通貨同士の差分」として導出する設計を
想定しています(例: `EURUSD_sentiment = EUR_sentiment - USD_sentiment`)。これは
ExpertTrader側で銘柄ごとに独立したスケーラー・分析モデルを使っている設計とも馴染みます。

## 使い方

```bash
pip install -r requirements.txt
python scripts/run_sentiment_pipeline.py --mode medium --lookback-hours 6
# -> sentiment_output/news_sentiment_medium.csv に1行追記される
```

cron(VPS)やタスクスケジューラ(ローカルPC)で定期実行する想定(例: 1時間ごと)。
`--lookback-hours`は実行間隔と同じかそれより少し長めに設定し、取りこぼしを防ぐ。

## 未決定事項・今後の課題

- RSSフィードの実際の疎通確認(上記「ニュース取得元」の注記を参照)。
- `currency_tags.py`のキーワード一致は簡易的なルールベース実装。誤検出・検出漏れが
  あれば、キーワードリストの調整、または将来的にはNER(固有表現抽出)等への
  置き換えを検討する。
- ExpertTrader本体との実際の統合(本リポジトリのCSV出力を、ExpertTrader側の
  `PortfolioFXEnv`の観測ベクトルにどう組み込むか)は未着手(条件5の「将来的な
  組み合わせ」に該当。今回は受け入れやすい出力形式の設計までが範囲)。
