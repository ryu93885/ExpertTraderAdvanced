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
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import feedparser

logger = logging.getLogger(__name__)

# 💡 収集対象のRSSフィード一覧。(source_label, url) のタプルで管理する。
# 疎通確認が取れていない場合に備え、1つのフィードが失敗しても他のフィードの
# 取得は続行する(_parse_feed内でtry/exceptしている)。
RSS_FEEDS = [
    ("ForexLive", "https://www.forexlive.com/feed/news"),
    ("DailyFX", "https://www.dailyfx.com/feeds/all"),
    ("Investing.com - Forex News", "https://www.investing.com/rss/news_285.rss"),
    ("FXStreet - News", "https://www.fxstreet.com/rss/news"),
]


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
            parsed = feedparser.parse(url)
        except Exception as e:
            logger.warning(f"⚠️ [{source_label}] フィード取得に失敗しました({url}): {e}")
            return []

        if getattr(parsed, "bozo", False) and not parsed.entries:
            logger.warning(f"⚠️ [{source_label}] フィードの解析に失敗した可能性があります({url}): {getattr(parsed, 'bozo_exception', '')}")
            return []

        results = []
        for entry in parsed.entries:
            published = self._extract_published(entry)
            if published is None or published < since:
                continue
            results.append(NewsArticle(
                title=entry.get("title", "").strip(),
                summary=entry.get("summary", entry.get("description", "")).strip(),
                published=published,
                source=source_label,
                url=entry.get("link", ""),
            ))
        return results

    @staticmethod
    def _extract_published(entry) -> Optional[datetime]:
        # feedparserは published_parsed / updated_parsed をtime.struct_timeで提供する
        for key in ("published_parsed", "updated_parsed"):
            t = entry.get(key)
            if t:
                return datetime(*t[:6], tzinfo=timezone.utc)
        return None
