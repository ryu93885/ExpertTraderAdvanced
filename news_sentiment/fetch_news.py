"""
ニュース記事の収集レイヤー。

設計方針(ユーザーとの合意: 「RSSフィードを使用し、将来的に簡単にほかのサービスに
移行できる仕様が望ましい」):
    収集元ごとの実装を NewsSource という共通インターフェースの背後に隠す。
    現時点では無料で使えるRSSフィード(RSSNewsSource)のみを実装するが、将来
    NewsAPI等の商用APIに切り替えたくなった場合は、NewsSourceを継承した新しい
    クラス(例: NewsApiSource)を追加するだけでよく、fetch_news.py より上位の
    レイヤー(aggregate.py・run_sentiment_pipeline.py)は一切変更不要になる。

💡 注意: ここに列挙しているRSS URLは、一般に公開されている主要な金融・FX系
ニュースサイトのフィードだが、サイト側の都合でURLやフィード提供自体が変わる
ことがある。本セッション(Claude Codeのサンドボックス環境)はネットワークポリシー
上、これらの外部サイトに直接接続できず疎通確認ができていないため、実際に
ローカルPC/VPSで実行する際に、必ず一度動作確認してください。フィードが無効化
されていた場合はRSS_FEEDSのリストから該当URLを削除/差し替えるだけでよい。

💡 追加(2026-11-25): GDELT Project(https://www.gdeltproject.org/)のDOC 2.0 APIを
使った GDELTNewsSource を追加した。RSSが「直近の記事しか取得できず、過去に遡れない」
弱点を補う目的で、過去の記事を遡及収集(バックフィル)する用途を主眼にしている。
ただし、GDELTの公式ドキュメントページ自体にも、本セッションのネットワークポリシー上
直接接続できず最終確認ができていない(RSS_FEEDS同様の制約)。パラメータ名・日付形式・
1回のリクエストで遡れる最大期間・レート制限は、複数の公開情報を基にした現時点での
最善の理解であり、実際にローカルPC/VPSで実行する際に必ず一度動作確認してください。
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import feedparser
import requests

logger = logging.getLogger(__name__)

# 💡 収集対象のRSSフィード一覧。(source_label, url) のタプルで管理する。
# 疎通確認が取れていない場合に備え、1つのフィードが失敗しても他のフィードの
# 取得は続行する(_parse_feed内でtry/exceptしている)。
#
# 💡 修正(2026-11-07の実機検証で判明): 当初は以下の4フィードだったが、
# 診断ログ(newest_published)を追加した上での実行結果、以下の2件を除外した。
#   - DailyFX ("https://www.dailyfx.com/feeds/all"): ブラウザ風ヘッダーを
#     付与しても status=403 で継続的にブロックされ、XMLとして「mismatched tag」
#     になる(典型的なブロックページの症状)。ヘッダー変更では解消しなかった。
#   - Investing.com ("https://www.investing.com/rss/news_285.rss"): status=200で
#     ブロックはされていないが、取得できる最新記事が2024-11-26と**約2年前**で
#     固定されていた。非ブラウザのクライアントに古い/静的なスナップショットを
#     返している(あるいは単純にフィード自体が更新停止している)とみられる。
# 残るForexLive/FXStreetは、同じ実機検証で「status=200/301・最新記事が実行時刻の
# 6〜8時間前」という正常な(週末でニュース自体が少なかっただけの)挙動を示しており、
# 問題なく機能している。別のソースを追加して冗長性を持たせたい場合は、実際に
# ブラウザで記事が新しいか確認した上でこのリストに追加すること。
RSS_FEEDS = [
    ("ForexLive", "https://www.forexlive.com/feed/news"),
    ("FXStreet - News", "https://www.fxstreet.com/rss/news"),
]

# 💡 追加(2026-11-07の実機検証で判明): feedparser.parse(url)をヘッダー無しで
# 呼ぶと、feedparser標準のUser-Agent(ブラウザではないと分かる文字列)がサイト側の
# bot対策(Cloudflare等)に弾かれ、RSS本来の内容の代わりにエラー/チャレンジページが
# 返ってくることがある。実機検証では4フィード中DailyFXでXMLパース失敗
# (「not well-formed」、典型的なブロックページ特有の症状)が発生し、残り3フィードも
# 含めて計0件という結果になった。実在のブラウザに近いヘッダーを明示的に付与する
# ことで、この種のブロックを避ける。
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
}


@dataclass
class NewsArticle:
    title: str
    summary: str
    published: datetime
    source: str
    url: str = ""

    @property
    def full_text(self) -> str:
        """FinBERTへの入力用に、タイトルと要約を結合したテキスト。"""
        if self.summary and self.summary != self.title:
            return f"{self.title}. {self.summary}"
        return self.title


class NewsSource(ABC):
    """
    ニュース収集元の共通インターフェース。将来的に別サービス(商用APIなど)に
    切り替える場合は、このクラスを継承した新しい実装を追加する。
    """

    @abstractmethod
    def fetch_recent(self, since: datetime) -> List[NewsArticle]:
        """since以降に公開された記事のリストを返す。"""
        raise NotImplementedError


class RSSNewsSource(NewsSource):
    """RSSフィードベースの実装(現時点での唯一の実装)。"""

    def __init__(self, feeds: Optional[List[tuple]] = None):
        self.feeds = feeds if feeds is not None else RSS_FEEDS

    def fetch_recent(self, since: datetime) -> List[NewsArticle]:
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)

        articles: List[NewsArticle] = []
        for source_label, url in self.feeds:
            articles.extend(self._parse_feed(source_label, url, since))
        logger.info(f"📰 RSS取得完了: {len(self.feeds)}フィードから計{len(articles)}件(since={since})")
        return articles

    def _parse_feed(self, source_label: str, url: str, since: datetime) -> List[NewsArticle]:
        try:
            parsed = feedparser.parse(url, request_headers=REQUEST_HEADERS)
        except Exception as e:
            logger.warning(f"⚠️ [{source_label}] フィード取得に失敗しました({url}): {e}")
            return []

        # 💡 追加: HTTPステータスが取れる場合は常にログへ残す。bot対策等でブロック
        # されている場合、本文はパースできてもステータスが200以外(403/429等)に
        # なることが多く、「正常に0件だった」のか「ブロックされて0件だった」のかを
        # 後からログだけで区別できるようにする。
        status = getattr(parsed, "status", None)

        if getattr(parsed, "bozo", False) and not parsed.entries:
            logger.warning(
                f"⚠️ [{source_label}] フィードの解析に失敗した可能性があります"
                f"(status={status}, {url}): {getattr(parsed, 'bozo_exception', '')}"
            )
            return []

        raw_count = len(parsed.entries)
        results = []
        newest_published = None  # 💡 追加: 対象期間外で捨てた記事も含め、取得できた
        # 記事の中で最も新しい日時を記録する。これにより「本当にすべて古かった」のか
        # 「日時のパース自体に失敗している(Noneのまま)」のかをログだけで切り分けられる。
        for entry in parsed.entries:
            published = self._extract_published(entry)
            if published is not None and (newest_published is None or published > newest_published):
                newest_published = published
            if published is None or published < since:
                continue
            results.append(NewsArticle(
                title=entry.get("title", "").strip(),
                summary=entry.get("summary", entry.get("description", "")).strip(),
                published=published,
                source=source_label,
                url=entry.get("link", ""),
            ))

        # 💡 追加: 以前はここで何もログを残していなかったため、「本当に新着が無い」
        # のか「ブロック等で空の応答が返ってきている」のかがログから区別できなかった。
        if raw_count == 0:
            logger.warning(f"⚠️ [{source_label}] 取得した記事が0件でした(status={status}, {url})。"
                            f"サイト側でブロックされていないか確認してください")
        elif len(results) == 0:
            # 💡 対象期間内の記事が0件でも、raw_count>0なら異常ではない可能性が高い
            # (newest_publishedがsinceのすぐ外側なら「本当に新着が無かった」だけ、
            # Noneのままなら日時のパース自体に失敗している可能性が高い)。
            logger.info(
                f"  [{source_label}] status={status}, 取得{raw_count}件中0件が対象期間内"
                f"(最新記事の日時={newest_published}, since={since})"
            )
        else:
            logger.info(f"  [{source_label}] status={status}, 取得{raw_count}件中{len(results)}件が対象期間内")

        return results

    @staticmethod
    def _extract_published(entry) -> Optional[datetime]:
        # feedparserは published_parsed / updated_parsed をtime.struct_timeで提供する
        for key in ("published_parsed", "updated_parsed"):
            t = entry.get(key)
            if t:
                return datetime(*t[:6], tzinfo=timezone.utc)
        return None


# 💡 GDELTのクエリ構文(ブール演算子・フレーズ検索)で、FX/マクロ経済に関連しそうな
# 記事を広めに拾う。厳密な通貨の絞り込みはcurrency_tags.py側で行うため、ここでは
# 「FX/マクロ関連かどうか」の大まかな事前フィルタとして機能すればよい。
DEFAULT_GDELT_QUERY = (
    '(forex OR "foreign exchange" OR "exchange rate" OR "interest rate" OR '
    '"central bank" OR "federal reserve" OR inflation OR "non-farm payroll")'
)


class GDELTNewsSource(NewsSource):
    """
    GDELT Project (https://www.gdeltproject.org/) のDOC 2.0 API
    (https://api.gdeltproject.org/api/v2/doc/doc) を使った実装。

    RSSNewsSourceと違い、startdatetime/enddatetimeを指定して任意の過去の期間を
    検索できる(fetch_range参照)。これにより、RSSでは不可能だった記事の遡及収集
    (バックフィル)が可能になる。ただし実際にどこまで遡れるか・1リクエストで
    返る最大件数は、GDELT API側の仕様変更に左右される(ファイル冒頭の注意書き参照)。

    💡 重要な制約: このAPIが返すのは記事のtitle・url・メタデータのみで、RSSの
    summary(要約本文)に相当するものは含まれない。そのためfull_textはtitleのみに
    なり、RSS由来の記事(title+summary)よりFinBERTへの入力情報がやや乏しくなる。
    これは意図的な設計判断であり(記事本文を毎回スクレイピングすると、DailyFX/
    Investing.comで経験したのと同種のブロック・サイト構造変化のリスクを再び
    抱えることになるため)、まずはtitleのみで実用上十分か検証し、不十分であれば
    本文取得の追加を検討する。
    """

    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
    # 💡 GDELT DOC APIの1リクエストあたりの最大取得件数(複数の公開情報を基にした
    # 想定値。実機で要確認)。この件数に達した場合、取りこぼしの可能性をログで警告する。
    MAX_RECORDS_PER_REQUEST = 250

    def __init__(self, query: str = DEFAULT_GDELT_QUERY, timeout: int = 30):
        self.query = query
        self.timeout = timeout

    def fetch_recent(self, since: datetime) -> List[NewsArticle]:
        """sinceから現在時刻までを取得する(定期実行での通常利用向け)。"""
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)
        return self.fetch_range(since, datetime.now(timezone.utc))

    def fetch_range(self, start: datetime, end: datetime) -> List[NewsArticle]:
        """
        指定した期間の記事を取得する(過去データの遡及収集=バックフィル向けに
        外部から直接呼ぶことを想定)。1リクエストで返せる件数に上限があるため、
        長い期間を一度に指定する場合は呼び出し側(バックフィルスクリプト)で
        期間を細かく分割して複数回呼ぶこと(このメソッド自体は単純に1期間=
        1リクエストのままにしている)。
        """
        params = {
            "query": self.query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": str(self.MAX_RECORDS_PER_REQUEST),
            "sort": "dateasc",
            "startdatetime": start.strftime("%Y%m%d%H%M%S"),
            "enddatetime": end.strftime("%Y%m%d%H%M%S"),
        }
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.warning(f"⚠️ [GDELT] 取得に失敗しました({start}〜{end}): {e}")
            return []

        raw_articles = data.get("articles", []) or []
        results = []
        skipped_non_english = 0
        for item in raw_articles:
            # 💡 FinBERT・currency_tags.pyはどちらも英語前提のため、英語以外の記事は
            # 除外する(クエリ側での言語指定もGDELT仕様の確証が取れていないため、
            # 念のためレスポンス側でも二重にチェックする)。
            language = (item.get("language") or "").strip().lower()
            if language and language not in ("english", "eng", "en"):
                skipped_non_english += 1
                continue
            published = self._parse_seendate(item.get("seendate"))
            if published is None:
                continue
            results.append(NewsArticle(
                title=(item.get("title") or "").strip(),
                summary="",  # 💡 GDELT DOC APIは記事本文・要約を提供しないため空のまま
                published=published,
                source=f"GDELT:{item.get('domain', 'unknown')}",
                url=item.get("url", ""),
            ))

        if len(raw_articles) >= self.MAX_RECORDS_PER_REQUEST:
            logger.warning(
                f"⚠️ [GDELT] {start}〜{end}の取得件数が上限({self.MAX_RECORDS_PER_REQUEST}件)に"
                "達した可能性があります。この期間に記事の取りこぼしがあるかもしれません"
                "(バックフィル側で期間をより細かく分割することを検討してください)。"
            )
        logger.info(
            f"📰 [GDELT] 取得完了: {len(results)}件({start}〜{end}、"
            f"非英語除外{skipped_non_english}件)"
        )
        return results

    @staticmethod
    def _parse_seendate(seendate: Optional[str]) -> Optional[datetime]:
        """GDELTのseendateは "20261124T120000Z" 形式(要実機確認)。"""
        if not seendate:
            return None
        try:
            return datetime.strptime(seendate, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            logger.warning(f"⚠️ [GDELT] seendateのパースに失敗しました: {seendate!r}")
            return None
