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
使えるRSSフィード(`RSSNewsSource`、`fetch_news.py`の`RSS_FEEDS`に列挙)のみを
実装しているが、将来的に商用ニュースAPI等へ切り替えたくなった場合は、`NewsSource`を
継承した新しいクラスを追加するだけでよく、`aggregate.py`以降のパイプラインは変更不要。

現在の`RSS_FEEDS`はForexLive・FXStreetの2件(2026-11-07の実機検証で、ブラウザ風
ヘッダーを付与した上でも継続的にブロックされる・取得内容が2年前で固定されている
ことが判明したDailyFX・Investing.comは除外した。詳細は`fetch_news.py`のコメント・
下記「動作確認済み事項」を参照)。別のソースを追加して冗長性を高めたい場合は、
実際にブラウザ等で記事が新しいことを確認した上で`RSS_FEEDS`に追加すること。

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
# -> sentiment_output/raw_articles_medium.csv に記事単位のデータが追記される(重複除外済み)
```

どのディレクトリから実行してもよい(スクリプト自身の位置からリポジトリルートを
自動的に解決する)。

cron(VPS)やタスクスケジューラ(ローカルPC)で定期実行する想定(例: 1時間ごと)。
`--lookback-hours`は実行間隔と同じかそれより少し長めに設定し、取りこぼしを防ぐ。
実行ログは`logs/run_sentiment_pipeline.log`に自動保存される(5MB×5世代でローテーション。
コンソール出力が残らないcron/タスクスケジューラ環境でも、常にファイルで確認できる)。

### 記事データの蓄積とバックアップ(--backup-dir)

RSSフィードは直近の記事しか取得できず、一度流れた記事は再取得できないため、
記事単位のデータ(`raw_articles_{mode}.csv`: title・summary・url・source・published・
currency・compound_sentiment)を重複除外しながら蓄積している。将来、FinBERTの
差し替えやキーワード判定(`currency_tags.py`)の改善を行う際、過去の記事に対して
再計算するために必要。

ローカルPC/VPS運用では、ローカル・GitHub・Google Drive間の手動操作を完全には
無くせないため、`--backup-dir`で複製先ディレクトリを指定すると、出力ファイル
(集計結果CSV・記事単位CSV)を**プログラム内で2か所に書き込む**(既存の
`rclone`等によるDrive同期フォルダ配下を指定すれば、Driveへのバックアップも
自動的に行われる)。省略時は従来通り1か所のみへの保存。

```bash
python scripts/run_sentiment_pipeline.py --mode medium --lookback-hours 6 \
    --backup-dir "C:\path\to\rclone-synced-folder"
```

⚠️ 2026-11-10時点では、`ExpertTraderAdvanced`自体はまだrcloneの同期対象に入っていない
(`ExpertTrader`配下の`datafetch_modeltrain`のみが同期対象)。そのため、当面は
`--backup-dir`に`ExpertTrader`側の同期済みフォルダ配下の専用サブフォルダ
(例: `...\ExpertTrader\datafetch_modeltrain\sentiment_backup`)を指定することで、
rcloneの設定変更なしにバックアップを機能させられる。`ExpertTraderAdvanced`自体を
rcloneの同期対象に加えれば、そちらを指定する形に変更してよい。

## 定期実行の設定

### cron (VPS / Linuxサーバー)

`crontab -e`で以下のように登録する(例: 毎時5分に、medium モードを過去1時間分の
ニュースで実行):

```cron
5 * * * * cd /path/to/ExpertTraderAdvanced && /path/to/venv/bin/python scripts/run_sentiment_pipeline.py --mode medium --lookback-hours 1.5 --backup-dir /path/to/rclone-synced-folder >> logs/cron_stdout.log 2>&1
```

- `cd`でリポジトリルートに移動してから実行する(`sentiment_output/`・`logs/`は
  カレントディレクトリ基準の相対パスで作られるため)。
- Pythonは仮想環境のものを絶対パスで指定するのが確実(cronは`PATH`が最小限の
  環境で動くため、`python`や`python3`だけだと意図したインタプリタに届かないことがある)。
- `--lookback-hours`は実行間隔(ここでは1時間)より少し長め(1.5時間)にして、
  cronの起動遅延等による取りこぼしを防ぐ。
- `>> logs/cron_stdout.log 2>&1`は保険。通常のログは上記の通りスクリプト自身が
  `logs/run_sentiment_pipeline.log`に記録するので無くても動くが、万一スクリプト
  起動前の段階(Pythonが見つからない等)で失敗した場合はこちらにしか残らない。
- longモードも併用する場合は、同じ書式で2行目を追加する(`--mode long`)。

### タスクスケジューラ (ローカルPC / Windows)

1. 「タスクスケジューラ」を開き、「タスクの作成」を選択
2. 「全般」タブ: 名前を入力(例: `ExpertTraderAdvanced_sentiment`)。「ユーザーが
   ログオンしているかどうかにかかわらず実行する」を選ぶとPC起動中は確実に動く
3. 「トリガー」タブ: 新規作成 →「毎日」→ 詳細設定で「繰り返し間隔: 1時間」
   「継続時間: 無期限」に設定
4. 「操作」タブ: 新規作成 →
   - プログラム/スクリプト: 仮想環境の`python.exe`へのフルパス
     (例: `C:\path\to\venv\Scripts\python.exe`)
   - 引数の追加: `scripts\run_sentiment_pipeline.py --mode medium --lookback-hours 1.5 --backup-dir "C:\path\to\rclone-synced-folder"`
   - 開始(作業)フォルダー: リポジトリのルートフォルダ(例: `C:\path\to\ExpertTraderAdvanced`)
     ← これを必ず設定する。空のままだとスクリプトが相対パスで探す
     `sentiment_output/`・`logs/`が別の場所に作られてしまう
5. 保存後、一度「実行」で手動起動して`logs/run_sentiment_pipeline.log`に
   記録が残ることを確認する

### 複数回分の出力・ログの確認方法

- センチメントの集計結果: `sentiment_output/news_sentiment_{mode}.csv`に1実行=1行で
  追記され続ける。`time`列で時系列に並んでいるので、Excel/pandasで開いて
  推移を確認できる。
- 記事単位のデータ: `sentiment_output/raw_articles_{mode}.csv`に、重複除外した
  記事が追記され続ける。
- 実行ログ: `logs/run_sentiment_pipeline.log`(最新)。featureごとの記事取得件数や
  warning(例: 特定フィードの取得失敗、対象期間の記事0件)もここに残る。
  特に立ち上げ直後は、記事件数が極端に少ない/0が続いていないか数回分を見て、
  RSSフィードが継続的に機能しているか確認するとよい。

## 未決定事項・今後の課題

- `currency_tags.py`のキーワード一致は簡易的なルールベース実装。誤検出・検出漏れが
  あれば、キーワードリストの調整、または将来的にはNER(固有表現抽出)等への
  置き換えを検討する(2026-11-06の実機検証では、1回の実行でAUD関連記事が0件になる
  ケースを確認している。複数回分のログを見て傾向を判断する)。
- ExpertTrader本体との実際の統合(本リポジトリのCSV出力を、ExpertTrader側の
  `PortfolioFXEnv`の観測ベクトルにどう組み込むか)は未着手(条件5の「将来的な
  組み合わせ」に該当。今回は受け入れやすい出力形式の設計までが範囲)。

## 動作確認済み事項

- 2026-11-06: ローカルPCでの実機実行により、RSSフィード(`fetch_news.py`の
  `RSS_FEEDS`)からの取得・FinBERTでのセンチメント推論・通貨ペア別センチメントの
  導出・CSV出力までの一連のパイプラインが正しく動作することを確認済み。
  7銘柄・5通貨の差分関係(例: GBPJPY = GBPUSD由来のGBP成分 + USDJPY由来の
  (−USD→JPY)成分)が実データ上でも数値的に矛盾なく成立していることも検証済み。
- 2026-11-07: 週末(土曜)にニュースが少ない状況での複数回の再実行と、
  各フィードの最新記事日時を出すログを追加した上での切り分けにより、
  ForexLive・FXStreetは正常に機能している(最新記事が実行時刻の6〜8時間前
  程度で、週末のニュース減少による一時的なものと判断)ことを確認。一方で
  DailyFXは継続的にブロックされ(status=403)、Investing.comは取得内容が
  2年前の記事で固定されている(status=200だが実質的に死んでいるフィード)
  ことが判明したため、この2件を`RSS_FEEDS`から除外した。
- 2026-11-10: 市場再開後の実行で、2フィードから実際に記事が取得できることを
  再確認(週末の記事減少が原因だったという前回の判断が裏付けられた)。また、
  記事単位データの蓄積機能(`append_raw_articles`)を追加し、重複除外ロジック
  (実機と同じ形の検証データで、新規記事のみが追記されること・CSV往復後の
  日時比較でも正しく重複検出できることを確認)。
