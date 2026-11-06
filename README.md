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

## アーキテクチャ(現時点の設計案)

```
news_sentiment/
  sentiment_model.py   … FinBERTのロードと推論(実装済み)
  (fetch_news.py)       … ニュース記事の収集(ソース未確定、下記「未決定事項」参照)
  (aggregate.py)         … 銘柄/通貨ごとのセンチメントスコア集計・CSV出力
scripts/
  (run_sentiment_pipeline.py) … 上記を定期実行するエントリーポイント(cron/タスクスケジューラ想定)
```

### 通貨ペア vs 通貨単位での集計について

ExpertTraderの対象7銘柄(GBPUSD, EURUSD, USDJPY, AUDUSD, GBPJPY, EURJPY, GOLD)は、
いずれも2つの通貨(またはGOLD)の組み合わせです。「USD利上げ」のようなニュースは
USDJPY・EURUSD・AUDUSD・GBPUSD全てに同時に影響しますが、影響の向きはペアによって
異なります(USD高ニュース → USDJPY上昇要因だがEURUSD下落要因)。

そのため、ニュースのセンチメントは**通貨単位(USD/EUR/GBP/JPY/AUD + GOLD/地政学)**で
集計し、各通貨ペアのセンチメント特徴量は「構成通貨同士の差分」として導出する設計を
想定しています(例: `EURUSD_sentiment = EUR_sentiment - USD_sentiment`)。これは
ExpertTrader側で銘柄ごとに独立したスケーラー・分析モデルを使っている設計とも馴染みます。

## 未決定事項(要相談)

**ニュースデータの取得元が未確定です。** 候補:
- NewsAPI.org等の商用ニュースAPI(無料枠あり、要APIキー、レート制限あり)
- Reuters/ForexLive/DailyFX等のRSSフィード(無料・APIキー不要だが、利用規約の確認が必要)
- 経済指標カレンダー系API(ForexFactory等。個別ニュース記事というより指標発表イベント)

既にお持ちのサービス・契約があればそれに合わせて実装します。特になければ、まずは
無料で試せるRSSベースの構成から始めることを提案します。
