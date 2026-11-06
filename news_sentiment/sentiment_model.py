"""
経済ニュース記事の文章から金融センチメント(ポジティブ/ネガティブ/ニュートラル)を
推論するモデルのラッパー。

設計方針(条件3: 転移学習のみで使用できるモデルにすること):
    ProsusAI/finbert (Hugging Face上で公開されている、金融テキストで事前学習・
    ファインチューニング済みのBERTベースモデル)をそのまま使う。本プロジェクトでは
    このモデル自体をゼロから学習することはしない。将来的に自前のラベル付きデータで
    精度を追い込みたくなった場合も、ベースのTransformer本体は凍結し、出力層
    (分類ヘッド)のみを追加で転移学習する前提とする(ExpertTrader側のCNN分析
    モデルを再学習しないのと同じ考え方)。

ローカルPC・VPSでの実行(条件4)を想定し、GPUが無い環境でもCPU推論で動作する
(FinBERTはBERT-baseサイズ(約1.1億パラメータ)で、定期バッチ推論程度の用途なら
CPUでも現実的な時間で動く)。
"""

import logging
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

MODEL_NAME = "ProsusAI/finbert"
# 💡 FinBERTのラベル順序は固定(positive, negative, neutral)。
# モデルのconfig.id2labelから動的に取得することで、将来モデルを差し替えても
# ラベル対応がズレないようにする。
_LABEL_ORDER_FALLBACK = ["positive", "negative", "neutral"]


class FinBertSentimentModel:
    """
    ProsusAI/finbertのロードと、テキストバッチに対するセンチメントスコア推論を行う。
    使い方:
        model = FinBertSentimentModel()
        results = model.score_texts(["Fed raises interest rates...", "..."])
        # results[0] = {"positive": 0.05, "negative": 0.82, "neutral": 0.13}
    """

    def __init__(self, device: str = None, batch_size: int = 16, max_length: int = 256):
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.batch_size = batch_size
        self.max_length = max_length

        logger.info(f"FinBERTモデルをロード中... (device={self.device})")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

        id2label = getattr(self.model.config, "id2label", None)
        if id2label:
            self.label_order = [id2label[i].lower() for i in range(len(id2label))]
        else:
            self.label_order = _LABEL_ORDER_FALLBACK
        logger.info(f"✅ FinBERTモデルのロード完了。ラベル順: {self.label_order}")

    @torch.no_grad()
    def score_texts(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        テキストのリストを受け取り、各テキストについて
        {"positive": float, "negative": float, "neutral": float} の確率辞書を返す。
        空文字列・空リストは安全に0件として扱う。
        """
        if not texts:
            return []

        results: List[Dict[str, float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", truncation=True,
                padding=True, max_length=self.max_length,
            ).to(self.device)

            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            for row in probs:
                results.append({label: float(p) for label, p in zip(self.label_order, row)})

        return results

    @staticmethod
    def to_compound_score(prob_dict: Dict[str, float]) -> float:
        """
        {"positive", "negative", "neutral"} の確率辞書を、-1(強いネガティブ)〜
        +1(強いポジティブ)の単一スカラーに圧縮する(positive - negative)。
        neutralが高いテキストは自然と0に近づく。ExpertTrader側の観測ベクトルに
        1次元の特徴量として組み込む場合など、単一の数値が欲しい場面向けの補助関数。
        """
        return prob_dict.get("positive", 0.0) - prob_dict.get("negative", 0.0)
