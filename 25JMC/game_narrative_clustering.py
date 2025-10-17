#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ゲームマニュアル物語記述のクラスタリング分析スクリプト

前処理済みデータを使用して、ゲームごとの物語に関する記述を
BERTエンコーディング、次元削減、クラスタリングにより分析する。
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

# 機械学習ライブラリ
try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learnが利用できません。")

# 次元削減ライブラリ
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("umap-learnが利用できません。")

# BERT関連ライブラリ
try:
    from transformers import AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("transformersが利用できません。")

# 言語判定ライブラリ
try:
    import langdetect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("langdetectが利用できません。")

# 日本語フォント設定
try:
    import japanize_matplotlib
    plt.rcParams['axes.unicode_minus'] = False
except ImportError:
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GameNarrativeClustering:
    """ゲームマニュアル物語記述のクラスタリング分析クラス"""
    
    def __init__(self, 
                 bert_model_name: str = "cl-tohoku/bert-base-japanese",
                 max_length: int = 512,
                 use_sentence_transformer: bool = True):
        """
        Args:
            bert_model_name: 使用するBERTモデル名
            max_length: 最大トークン数
            use_sentence_transformer: SentenceTransformerを使用するかどうか
        """
        self.bert_model_name = bert_model_name
        self.max_length = max_length
        self.use_sentence_transformer = use_sentence_transformer
        
        # モデルとトークナイザー
        self.model = None
        self.tokenizer = None
        self.sentence_transformer = None
        
        # データ
        self.documents = []
        self.document_metadata = []
        self.embeddings = None
        
        # クラスタリング結果
        self.cluster_labels = None
        self.cluster_centers = None
        self.last_clustering_embeddings = None
        
        # 初期化
        self._initialize_models()

    @staticmethod
    def _to_serializable(obj):
        """JSONシリアライズ可能な形に再帰的に変換（numpy型や辞書キーの整形）"""
        import numpy as _np
        if isinstance(obj, dict):
            return {str(GameNarrativeClustering._to_serializable(k)): GameNarrativeClustering._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [GameNarrativeClustering._to_serializable(v) for v in obj]
        if isinstance(obj, tuple):
            return [GameNarrativeClustering._to_serializable(v) for v in obj]
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, (_np.ndarray,)):
            return obj.tolist()
        return obj
    
    def _initialize_models(self):
        """BERTモデルとトークナイザーを初期化"""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformersライブラリが利用できません")
            return
        
        try:
            if self.use_sentence_transformer:
                # SentenceTransformerを使用（推奨）
                logger.info(f"SentenceTransformerモデルを初期化: {self.bert_model_name}")
                self.sentence_transformer = SentenceTransformer(self.bert_model_name)
                logger.info("SentenceTransformer初期化完了")
            else:
                # 通常のBERTモデルを使用
                logger.info(f"BERTモデルを初期化: {self.bert_model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
                self.model = AutoModel.from_pretrained(self.bert_model_name)
                logger.info("BERTモデル初期化完了")
        except Exception as e:
            logger.error(f"モデル初期化エラー: {e}")
            # フォールバック: より軽量なモデル
            try:
                fallback_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                logger.info(f"フォールバックモデルを使用: {fallback_model}")
                self.sentence_transformer = SentenceTransformer(fallback_model)
                self.use_sentence_transformer = True
            except Exception as e2:
                logger.error(f"フォールバックモデルも失敗: {e2}")
    
    def load_data(self, input_file: str) -> List[Dict]:
        """データの読み込みと前処理"""
        logger.info(f"データ読み込み: {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        metadata = []
        
        for manual in data.get('manuals', []):
            manual_title = manual.get('manual_title', '')
            game_id = manual.get('game_id', '')
            
            for section in manual.get('narrative_sections', []):
                section_id = section.get('id', '')
                section_head = section.get('head', '')
                paragraphs = section.get('paragraphs', [])
                
                # セクション全体のテキストを結合
                full_text = ' '.join(paragraphs)
                
                if full_text.strip():  # 空でない場合のみ追加
                    # 前処理
                    processed_text = self._preprocess_text(full_text)
                    
                    if processed_text:
                        documents.append(processed_text)
                        metadata.append({
                            'manual_title': manual_title,
                            'game_id': game_id,
                            'section_id': section_id,
                            'section_head': section_head,
                            'original_text': full_text,
                            'processed_text': processed_text
                        })
        
        self.documents = documents
        self.document_metadata = metadata
        
        logger.info(f"読み込み完了: {len(documents)}件のドキュメント")
        return documents
    
    def _preprocess_text(self, text: str) -> str:
        """テキストの前処理"""
        import unicodedata
        
        # 1. Unicode正規化（NFKC）
        text = unicodedata.normalize('NFKC', text)
        
        # 2. 改行・記号類の統一
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = text.replace('　', ' ')  # 全角スペースを半角に
        text = ' '.join(text.split())  # 空白の正規化
        
        # 3. 言語判定（簡易版）
        if LANGDETECT_AVAILABLE:
            try:
                lang = langdetect.detect(text)
                if lang != 'ja':
                    logger.warning(f"非日本語テキストを検出: {lang}")
            except:
                pass
        
        # 4. 長さチェックと分割
        if len(text) > self.max_length * 3:  # 概算でトークン数をチェック
            # 長すぎる場合は要約（簡易版）
            sentences = text.split('。')
            if len(sentences) > 3:
                text = '。'.join(sentences[:3]) + '。'
        
        return text.strip()
    
    def create_embeddings(self) -> np.ndarray:
        """BERTエンコーディング（文書ベクトル化）"""
        if not self.documents:
            logger.error("ドキュメントが読み込まれていません")
            return None
        
        logger.info("BERTエンコーディング開始")
        
        if self.use_sentence_transformer and self.sentence_transformer:
            # SentenceTransformerを使用（推奨）
            embeddings = self.sentence_transformer.encode(
                self.documents,
                batch_size=8,
                show_progress_bar=True,
                convert_to_numpy=True
            )
        elif self.model and self.tokenizer:
            # 通常のBERTモデルを使用
            embeddings = []
            
            for i, text in enumerate(self.documents):
                logger.info(f"エンコーディング中: {i+1}/{len(self.documents)}")
                
                # トークン化
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding=True
                )
                
                # モデル推論
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # mean pooling（トークンの平均）
                    attention_mask = inputs['attention_mask']
                    embeddings_tensor = outputs.last_hidden_state
                    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings_tensor.size()).float()
                    sum_embeddings = torch.sum(embeddings_tensor * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    embedding = sum_embeddings / sum_mask
                    embeddings.append(embedding.squeeze().numpy())
            
            embeddings = np.array(embeddings)
        else:
            logger.error("有効なモデルがありません")
            return None
        
        self.embeddings = embeddings
        logger.info(f"エンコーディング完了: {embeddings.shape}")
        return embeddings
    
    def reduce_dimensions(self, 
                         method: str = 'umap',
                         n_components: int = 2,
                         use_pca: bool = True,
                         pca_components: int = 100,
                         umap_neighbors: Optional[int] = None,
                         umap_min_dist: float = 0.1) -> np.ndarray:
        """次元削減"""
        if self.embeddings is None:
            logger.error("エンベディングが作成されていません")
            return None
        
        logger.info(f"次元削減開始: {method}")
        
        # PCAによる前処理（オプション）
        n_samples, n_features = self.embeddings.shape
        embeddings_pca = self.embeddings
        if use_pca:
            # n_components は min(n_samples-1, n_features, pca_components) 以下である必要がある
            max_components = max(2, min(n_samples - 1, n_features))
            pca_n = min(pca_components, max_components)
            if pca_n < n_features:
                logger.info(f"PCA前処理: {n_features} -> {pca_n}")
                pca = PCA(n_components=pca_n, random_state=42)
                embeddings_pca = pca.fit_transform(self.embeddings)
                try:
                    logger.info(f"PCA説明分散比: {float(np.sum(pca.explained_variance_ratio_)):.3f}")
                except Exception:
                    pass
        
        # 次元削減
        if method == 'umap' and UMAP_AVAILABLE:
            logger.info("UMAP次元削減実行")
            # n_neighbors は 2 以上 n_samples-1 以下
            if umap_neighbors is None:
                umap_neighbors = max(2, min(15, n_samples - 1))
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=umap_neighbors,
                min_dist=umap_min_dist,
                metric='cosine',
                random_state=42
            )
            reduced_embeddings = reducer.fit_transform(embeddings_pca)
        
        elif method == 'tsne':
            logger.info("t-SNE次元削減実行")
            tsne = TSNE(
                n_components=n_components,
                perplexity=min(30, len(embeddings_pca) - 1),
                random_state=42
            )
            reduced_embeddings = tsne.fit_transform(embeddings_pca)
        
        elif method == 'pca':
            logger.info("PCA次元削減実行")
            pca = PCA(n_components=n_components, random_state=42)
            reduced_embeddings = pca.fit_transform(embeddings_pca)
        
        else:
            logger.warning(f"指定された方法 {method} は利用できません。PCAを使用します")
            pca = PCA(n_components=n_components, random_state=42)
            reduced_embeddings = pca.fit_transform(embeddings_pca)
        
        logger.info(f"次元削減完了: {reduced_embeddings.shape}")
        return reduced_embeddings
    
    def perform_clustering(self, 
                          method: str = 'hdbscan',
                          n_clusters: int = 5,
                          embeddings: Optional[np.ndarray] = None) -> np.ndarray:
        """クラスタリング実行"""
        if embeddings is None:
            embeddings = self.embeddings
        
        if embeddings is None:
            logger.error("エンベディングが利用できません")
            return None
        
        logger.info(f"クラスタリング開始: {method}")
        
        if method == 'kmeans':
            logger.info(f"KMeansクラスタリング: {n_clusters}クラスタ")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            self.cluster_centers = kmeans.cluster_centers_
        
        elif method == 'dbscan':
            logger.info("DBSCANクラスタリング")
            # データを標準化
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings)
            
            dbscan = DBSCAN(eps=0.5, min_samples=3)
            cluster_labels = dbscan.fit_predict(embeddings_scaled)
        
        elif method == 'hdbscan' and UMAP_AVAILABLE:
            try:
                from hdbscan import HDBSCAN
                logger.info("HDBSCANクラスタリング")
                # 少ノイズ・多クラスタ化を狙ってパラメータ緩和
                hdbscan = HDBSCAN(
                    min_cluster_size=3,
                    min_samples=1,
                    metric='euclidean',
                    cluster_selection_method='eom'
                )
                cluster_labels = hdbscan.fit_predict(embeddings)
            except ImportError:
                logger.warning("HDBSCANが利用できません。DBSCANを使用します")
                dbscan = DBSCAN(eps=0.5, min_samples=3)
                cluster_labels = dbscan.fit_predict(embeddings)
        
        else:
            logger.warning(f"指定された方法 {method} は利用できません。KMeansを使用します")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            self.cluster_centers = kmeans.cluster_centers_
        
        self.cluster_labels = cluster_labels
        # 代表抽出などで使用するため、クラスタリングに使った埋め込みを保存
        self.last_clustering_embeddings = embeddings
        
        # クラスタリング評価
        self._evaluate_clustering(embeddings, cluster_labels)
        
        logger.info(f"クラスタリング完了: {len(set(cluster_labels))}クラスタ")
        return cluster_labels
    
    def _evaluate_clustering(self, embeddings: np.ndarray, labels: np.ndarray):
        """クラスタリングの評価"""
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters > 1:
            try:
                # シルエットスコア
                silhouette_avg = silhouette_score(embeddings, labels)
                logger.info(f"シルエットスコア: {silhouette_avg:.3f}")
                
                # Calinski-Harabaszスコア
                calinski_score = calinski_harabasz_score(embeddings, labels)
                logger.info(f"Calinski-Harabaszスコア: {calinski_score:.3f}")
            except Exception as e:
                logger.warning(f"クラスタリング評価エラー: {e}")
    
    def find_optimal_clusters_elbow(self, 
                                   embeddings: Optional[np.ndarray] = None,
                                   max_clusters: int = 15,
                                   min_clusters: int = 2) -> Tuple[int, Dict]:
        """エルボー法による最適クラスタ数の決定"""
        if embeddings is None:
            embeddings = self.embeddings
        
        if embeddings is None:
            logger.error("エンベディングが利用できません")
            return None, {}
        
        logger.info(f"エルボー法による最適クラスタ数探索開始 ({min_clusters}-{max_clusters})")
        
        # データ数に応じてmax_clustersを調整
        n_samples = embeddings.shape[0]
        max_clusters = min(max_clusters, n_samples - 1)
        
        inertias = []
        k_range = range(min_clusters, max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(embeddings)
            inertias.append(kmeans.inertia_)
            logger.info(f"k={k}: inertia={kmeans.inertia_:.2f}")
        
        # エルボーポイント検出（差分の差分が最大になる点）
        if len(inertias) >= 3:
            # 1次差分
            first_diffs = np.diff(inertias)
            # 2次差分（変化率の変化）
            second_diffs = np.diff(first_diffs)
            # 最大の変化率変化を示すポイント（エルボー）
            elbow_idx = np.argmax(second_diffs) + 2  # +2 because of double diff
            optimal_k = k_range[elbow_idx] if elbow_idx < len(k_range) else k_range[-1]
        else:
            optimal_k = min_clusters
        
        results = {
            'k_range': list(k_range),
            'inertias': inertias,
            'optimal_k': optimal_k,
            'method': 'elbow'
        }
        
        logger.info(f"エルボー法による最適クラスタ数: {optimal_k}")
        return optimal_k, results
    
    def find_optimal_clusters_silhouette(self, 
                                        embeddings: Optional[np.ndarray] = None,
                                        max_clusters: int = 15,
                                        min_clusters: int = 2) -> Tuple[int, Dict]:
        """シルエット分析による最適クラスタ数の決定"""
        if embeddings is None:
            embeddings = self.embeddings
        
        if embeddings is None:
            logger.error("エンベディングが利用できません")
            return None, {}
        
        logger.info(f"シルエット分析による最適クラスタ数探索開始 ({min_clusters}-{max_clusters})")
        
        # データ数に応じてmax_clustersを調整
        n_samples = embeddings.shape[0]
        max_clusters = min(max_clusters, n_samples - 1)
        
        silhouette_scores = []
        k_range = range(min_clusters, max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            logger.info(f"k={k}: silhouette_score={silhouette_avg:.3f}")
        
        # 最高シルエットスコアを持つクラスタ数
        optimal_idx = np.argmax(silhouette_scores)
        optimal_k = k_range[optimal_idx]
        
        results = {
            'k_range': list(k_range),
            'silhouette_scores': silhouette_scores,
            'optimal_k': optimal_k,
            'max_score': silhouette_scores[optimal_idx],
            'method': 'silhouette'
        }
        
        logger.info(f"シルエット分析による最適クラスタ数: {optimal_k} (スコア: {silhouette_scores[optimal_idx]:.3f})")
        return optimal_k, results
    
    def find_optimal_clusters_combined(self, 
                                      embeddings: Optional[np.ndarray] = None,
                                      max_clusters: int = 15,
                                      min_clusters: int = 2) -> Tuple[int, Dict]:
        """複数手法を組み合わせた最適クラスタ数の決定"""
        if embeddings is None:
            embeddings = self.embeddings
        
        if embeddings is None:
            logger.error("エンベディングが利用できません")
            return None, {}
        
        logger.info("複数手法による最適クラスタ数探索開始")
        
        # 各手法で最適クラスタ数を求める
        elbow_k, elbow_results = self.find_optimal_clusters_elbow(embeddings, max_clusters, min_clusters)
        silhouette_k, silhouette_results = self.find_optimal_clusters_silhouette(embeddings, max_clusters, min_clusters)
        
        # 結果を統合
        all_ks = [elbow_k, silhouette_k]
        # 最頻値を採用（同じクラスタ数が複数の手法で選ばれた場合）
        from collections import Counter
        k_counts = Counter(all_ks)
        
        if len(k_counts) == 1:
            # 全て同じ値
            optimal_k = list(k_counts.keys())[0]
            confidence = "高"
        else:
            # 異なる値の場合、シルエットスコアを重視
            optimal_k = silhouette_k
            confidence = "中"
        
        results = {
            'elbow_k': elbow_k,
            'silhouette_k': silhouette_k,
            'optimal_k': optimal_k,
            'confidence': confidence,
            'elbow_results': elbow_results,
            'silhouette_results': silhouette_results,
            'method': 'combined'
        }
        
        logger.info(f"統合手法による最適クラスタ数: {optimal_k} (信頼度: {confidence})")
        logger.info(f"  - エルボー法: {elbow_k}")
        logger.info(f"  - シルエット法: {silhouette_k}")
        
        return optimal_k, results
    
    def plot_cluster_optimization(self, optimization_results: Dict, output_dir: Path):
        """クラスタ数最適化の可視化"""
        method = optimization_results.get('method', 'unknown')
        
        if method == 'elbow':
            self._plot_elbow_curve(optimization_results, output_dir)
        elif method == 'silhouette':
            self._plot_silhouette_analysis(optimization_results, output_dir)
        elif method == 'combined':
            self._plot_combined_optimization(optimization_results, output_dir)
    
    def _plot_elbow_curve(self, results: Dict, output_dir: Path):
        """エルボー曲線の可視化"""
        plt.figure(figsize=(10, 6))
        
        k_range = results['k_range']
        inertias = results['inertias']
        optimal_k = results['optimal_k']
        
        plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8, label='慣性（Inertia）')
        plt.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, 
                   label=f'最適クラスタ数 = {optimal_k}')
        
        plt.xlabel('クラスタ数 (k)', fontsize=12)
        plt.ylabel('慣性（Inertia）', fontsize=12)
        plt.title('エルボー法によるクラスタ数最適化', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'elbow_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_silhouette_analysis(self, results: Dict, output_dir: Path):
        """シルエット分析の可視化"""
        plt.figure(figsize=(10, 6))
        
        k_range = results['k_range']
        silhouette_scores = results['silhouette_scores']
        optimal_k = results['optimal_k']
        max_score = results['max_score']
        
        plt.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8, 
                label='シルエットスコア')
        plt.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2,
                   label=f'最適クラスタ数 = {optimal_k} (スコア: {max_score:.3f})')
        
        plt.xlabel('クラスタ数 (k)', fontsize=12)
        plt.ylabel('シルエットスコア', fontsize=12)
        plt.title('シルエット分析によるクラスタ数最適化', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'silhouette_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_combined_optimization(self, results: Dict, output_dir: Path):
        """統合最適化結果の可視化"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # エルボー曲線
        elbow_results = results['elbow_results']
        k_range = elbow_results['k_range']
        inertias = elbow_results['inertias']
        elbow_k = results['elbow_k']
        
        ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.axvline(x=elbow_k, color='blue', linestyle='--', linewidth=2,
                   label=f'エルボー法: k={elbow_k}')
        ax1.set_xlabel('クラスタ数 (k)')
        ax1.set_ylabel('慣性（Inertia）')
        ax1.set_title('エルボー法')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # シルエット分析
        silhouette_results = results['silhouette_results']
        silhouette_scores = silhouette_results['silhouette_scores']
        silhouette_k = results['silhouette_k']
        
        ax2.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
        ax2.axvline(x=silhouette_k, color='green', linestyle='--', linewidth=2,
                   label=f'シルエット法: k={silhouette_k}')
        ax2.set_xlabel('クラスタ数 (k)')
        ax2.set_ylabel('シルエットスコア')
        ax2.set_title('シルエット分析')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 総合結果を表示
        optimal_k = results['optimal_k']
        confidence = results['confidence']
        plt.suptitle(f'統合最適化結果: k={optimal_k} (信頼度: {confidence})', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'combined_optimization.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def extract_cluster_keywords(self, n_keywords: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """クラスタの代表語を抽出（TF-IDF）"""
        if self.cluster_labels is None:
            logger.error("クラスタリングが実行されていません")
            return {}
        
        logger.info("クラスタ代表語抽出開始")
        
        # TF-IDFベクトライザー
        tfidf = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        
        # TF-IDF行列作成
        tfidf_matrix = tfidf.fit_transform(self.documents)
        feature_names = tfidf.get_feature_names_out()
        
        # クラスタ別にキーワード抽出
        cluster_keywords = {}
        
        for cluster_id in set(self.cluster_labels):
            if cluster_id == -1:  # ノイズクラスタ
                continue
            
            # クラスタに属するドキュメントのインデックス
            cluster_indices = np.where(self.cluster_labels == cluster_id)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # クラスタ内のTF-IDF平均
            cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0).A1
            
            # 上位キーワードを取得
            top_indices = cluster_tfidf.argsort()[-n_keywords:][::-1]
            keywords = [(feature_names[i], cluster_tfidf[i]) for i in top_indices]
            
            cluster_keywords[cluster_id] = keywords
        
        logger.info("クラスタ代表語抽出完了")
        return cluster_keywords
    
    def find_cluster_representatives(self, n_representatives: int = 3) -> Dict[int, List[Dict]]:
        """クラスタの代表文書を抽出"""
        if self.cluster_labels is None or (self.embeddings is None and self.last_clustering_embeddings is None):
            logger.error("クラスタリングが実行されていません")
            return {}
        
        logger.info("クラスタ代表文書抽出開始")
        
        cluster_representatives = {}
        
        # 代表抽出に使う埋め込み（クラスタリングと同じ空間）
        base_embeddings = self.last_clustering_embeddings if self.last_clustering_embeddings is not None else self.embeddings

        for cluster_id in set(self.cluster_labels):
            if cluster_id == -1:  # ノイズクラスタ
                continue
            
            # クラスタに属するドキュメントのインデックス
            cluster_indices = np.where(self.cluster_labels == cluster_id)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # クラスタ内のエンベディング
            cluster_embeddings = base_embeddings[cluster_indices]
            
            if self.cluster_centers is not None and cluster_embeddings.shape[1] == self.cluster_centers.shape[1]:
                # クラスタ中心からの距離で代表文書を選択
                cluster_center = self.cluster_centers[cluster_id]
                distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
                closest_indices = np.argsort(distances)[:n_representatives]
            else:
                # クラスタ内の平均エンベディングからの距離
                cluster_mean = np.mean(cluster_embeddings, axis=0)
                distances = np.linalg.norm(cluster_embeddings - cluster_mean, axis=1)
                closest_indices = np.argsort(distances)[:n_representatives]
            
            representatives = []
            for idx in closest_indices:
                doc_idx = cluster_indices[idx]
                representatives.append({
                    'index': doc_idx,
                    'text': self.documents[doc_idx][:200] + '...' if len(self.documents[doc_idx]) > 200 else self.documents[doc_idx],
                    'metadata': self.document_metadata[doc_idx],
                    'distance': distances[idx]
                })
            
            cluster_representatives[cluster_id] = representatives
        
        logger.info("クラスタ代表文書抽出完了")
        return cluster_representatives
    
    def create_visualizations(self, reduced_embeddings: Optional[np.ndarray] = None):
        """可視化の作成"""
        logger.info("可視化作成開始")
        
        # 出力ディレクトリ作成
        output_dir = Path("narrative_clustering_results")
        output_dir.mkdir(exist_ok=True)
        
        # 1. クラスタ分布の可視化
        if reduced_embeddings is not None and self.cluster_labels is not None:
            self._plot_cluster_distribution(reduced_embeddings, output_dir)
        
        # 2. クラスタキーワードの可視化
        cluster_keywords = self.extract_cluster_keywords()
        if cluster_keywords:
            self._plot_cluster_keywords(cluster_keywords, output_dir)
        
        # 3. ゲームタイトル別クラスタ分布
        if self.cluster_labels is not None:
            self._plot_game_cluster_distribution(output_dir)
        
        # 4. クラスタ統計の可視化
        if self.cluster_labels is not None:
            self._plot_cluster_statistics(output_dir)
        
        logger.info("可視化作成完了")
    
    def _plot_cluster_distribution(self, reduced_embeddings: np.ndarray, output_dir: Path):
        """クラスタ分布の可視化"""
        plt.figure(figsize=(12, 10))
        
        # クラスタ別に色分け
        unique_labels = set(self.cluster_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            if label == -1:
                # ノイズクラスタ
                mask = self.cluster_labels == label
                plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1],
                           c='black', marker='x', s=50, alpha=0.6, label='ノイズ')
            else:
                mask = self.cluster_labels == label
                plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1],
                           c=[colors[i]], s=100, alpha=0.7, label=f'クラスタ {label}')
        
        plt.title('ゲームマニュアル物語記述のクラスタ分布', fontsize=16, fontweight='bold')
        plt.xlabel('次元1')
        plt.ylabel('次元2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'cluster_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cluster_keywords(self, cluster_keywords: Dict[int, List[Tuple[str, float]]], output_dir: Path):
        """クラスタキーワードの可視化"""
        n_clusters = len(cluster_keywords)
        if n_clusters == 0:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (cluster_id, keywords) in enumerate(cluster_keywords.items()):
            if i >= len(axes):
                break
            
            words, scores = zip(*keywords[:10])  # 上位10語
            
            axes[i].barh(range(len(words)), scores, color='lightblue', alpha=0.7)
            axes[i].set_yticks(range(len(words)))
            axes[i].set_yticklabels(words)
            axes[i].set_title(f'クラスタ {cluster_id} の代表語', fontweight='bold')
            axes[i].set_xlabel('TF-IDFスコア')
        
        # 未使用のサブプロットを非表示
        for i in range(len(cluster_keywords), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'cluster_keywords.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_game_cluster_distribution(self, output_dir: Path):
        """ゲームタイトル別クラスタ分布の可視化"""
        # ゲームタイトル別のクラスタ分布を集計
        game_cluster_counts = {}
        
        for i, metadata in enumerate(self.document_metadata):
            game_title = metadata['manual_title']
            cluster_id = self.cluster_labels[i]
            
            if game_title not in game_cluster_counts:
                game_cluster_counts[game_title] = {}
            
            if cluster_id not in game_cluster_counts[game_title]:
                game_cluster_counts[game_title][cluster_id] = 0
            
            game_cluster_counts[game_title][cluster_id] += 1
        
        # ヒートマップ作成
        games = list(game_cluster_counts.keys())
        clusters = sorted(set(self.cluster_labels))
        if -1 in clusters:  # ノイズクラスタを最後に
            clusters.remove(-1)
            clusters.append(-1)
        
        # データマトリックス作成
        data_matrix = []
        for game in games:
            row = []
            for cluster in clusters:
                row.append(game_cluster_counts[game].get(cluster, 0))
            data_matrix.append(row)
        
        plt.figure(figsize=(12, max(8, len(games) * 0.4)))
        sns.heatmap(data_matrix, 
                   xticklabels=[f'クラスタ {c}' if c != -1 else 'ノイズ' for c in clusters],
                   yticklabels=games,
                   annot=True, 
                   fmt='d',
                   cmap='YlOrRd',
                   cbar_kws={'label': 'ドキュメント数'})
        
        plt.title('ゲームタイトル別クラスタ分布', fontsize=16, fontweight='bold')
        plt.xlabel('クラスタ')
        plt.ylabel('ゲームタイトル')
        plt.tight_layout()
        plt.savefig(output_dir / 'game_cluster_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cluster_statistics(self, output_dir: Path):
        """クラスタ統計の可視化"""
        plt.figure(figsize=(15, 10))
        
        # クラスタサイズ分布
        plt.subplot(2, 2, 1)
        cluster_sizes = []
        cluster_labels_list = []
        
        for cluster_id in sorted(set(self.cluster_labels)):
            size = np.sum(self.cluster_labels == cluster_id)
            cluster_sizes.append(size)
            cluster_labels_list.append(f'クラスタ {cluster_id}' if cluster_id != -1 else 'ノイズ')
        
        bars = plt.bar(cluster_labels_list, cluster_sizes, color='lightcoral', alpha=0.7)
        plt.title('クラスタサイズ分布', fontweight='bold')
        plt.xlabel('クラスタ')
        plt.ylabel('ドキュメント数')
        plt.xticks(rotation=45)
        
        # バーの上に数値を表示
        for bar, size in zip(bars, cluster_sizes):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(size), ha='center', va='bottom')
        
        # ゲームタイトル別ドキュメント数
        plt.subplot(2, 2, 2)
        game_counts = {}
        for metadata in self.document_metadata:
            game_title = metadata['manual_title']
            game_counts[game_title] = game_counts.get(game_title, 0) + 1
        
        games = list(game_counts.keys())
        counts = list(game_counts.values())
        
        bars = plt.barh(games, counts, color='lightblue', alpha=0.7)
        plt.title('ゲームタイトル別ドキュメント数', fontweight='bold')
        plt.xlabel('ドキュメント数')
        
        # クラスタ内のゲーム多様性
        plt.subplot(2, 2, 3)
        cluster_diversity = []
        cluster_ids = []
        
        for cluster_id in sorted(set(self.cluster_labels)):
            if cluster_id == -1:
                continue
            
            cluster_indices = np.where(self.cluster_labels == cluster_id)[0]
            games_in_cluster = set()
            
            for idx in cluster_indices:
                games_in_cluster.add(self.document_metadata[idx]['manual_title'])
            
            cluster_diversity.append(len(games_in_cluster))
            cluster_ids.append(f'クラスタ {cluster_id}')
        
        if cluster_diversity:
            plt.bar(cluster_ids, cluster_diversity, color='lightgreen', alpha=0.7)
            plt.title('クラスタ内ゲーム多様性', fontweight='bold')
            plt.xlabel('クラスタ')
            plt.ylabel('ゲーム数')
            plt.xticks(rotation=45)
        
        # ドキュメント長分布
        plt.subplot(2, 2, 4)
        doc_lengths = [len(doc) for doc in self.documents]
        plt.hist(doc_lengths, bins=20, color='lightyellow', alpha=0.7, edgecolor='black')
        plt.title('ドキュメント長分布', fontweight='bold')
        plt.xlabel('文字数')
        plt.ylabel('頻度')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'cluster_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, reduced_embeddings: Optional[np.ndarray] = None):
        """結果の保存"""
        logger.info("結果保存開始")
        
        output_dir = Path("narrative_clustering_results")
        output_dir.mkdir(exist_ok=True)
        
        # 基本結果
        results = {
            'clustering_info': {
                'n_documents': len(self.documents),
                'n_clusters': len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0) if self.cluster_labels is not None else 0,
                'cluster_labels': self.cluster_labels.tolist() if self.cluster_labels is not None else None,
                'embedding_shape': self.embeddings.shape if self.embeddings is not None else None
            },
            'cluster_keywords': self.extract_cluster_keywords(),
            'cluster_representatives': self.find_cluster_representatives(),
            'document_metadata': self.document_metadata
        }
        # シリアライズ可能に整形
        results = GameNarrativeClustering._to_serializable(results)

        with open(output_dir / 'clustering_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # クラスタ別ドキュメントリスト
        if self.cluster_labels is not None:
            cluster_docs = {}
            for i, cluster_id in enumerate(self.cluster_labels):
                if cluster_id not in cluster_docs:
                    cluster_docs[cluster_id] = []
                
                cluster_docs[cluster_id].append({
                    'index': i,
                    'text': self.documents[i][:200] + '...' if len(self.documents[i]) > 200 else self.documents[i],
                    'metadata': self.document_metadata[i]
                })
            
            # シリアライズ可能に整形（キーを文字列化）
            cluster_docs = GameNarrativeClustering._to_serializable(cluster_docs)
            with open(output_dir / 'cluster_documents.json', 'w', encoding='utf-8') as f:
                json.dump(cluster_docs, f, ensure_ascii=False, indent=2)
        
        # CSVファイル
        if self.cluster_labels is not None:
            df_results = pd.DataFrame({
                'index': range(len(self.documents)),
                'cluster': self.cluster_labels,
                'text': [doc[:200] + '...' if len(doc) > 200 else doc for doc in self.documents],
                'manual_title': [meta['manual_title'] for meta in self.document_metadata],
                'game_id': [meta['game_id'] for meta in self.document_metadata],
                'section_head': [meta['section_head'] for meta in self.document_metadata]
            })
            
            df_results.to_csv(output_dir / 'clustering_results.csv', index=False, encoding='utf-8')
        
        logger.info("結果保存完了")
    
    def run_analysis(self, 
                    input_file: str = 'cleaned_data/cleaned_narrative_sections.json',
                    clustering_method: str = 'hdbscan',
                    n_clusters: Optional[int] = None,
                    auto_optimize_clusters: bool = True,
                    cluster_optimization_method: str = 'combined',
                    dimension_reduction: str = 'umap',
                    use_pca: bool = True):
        """メイン分析処理"""
        logger.info("=== ゲームマニュアル物語記述クラスタリング分析開始 ===")
        
        # 1. データ読み込み
        self.load_data(input_file)
        
        # 2. BERTエンコーディング
        self.create_embeddings()
        
        # 3. 次元削減
        # 低次元(10次元)に圧縮した埋め込みも作成（HDBSCANの安定化用）
        reduced_embeddings_10d = None
        if dimension_reduction == 'umap' and UMAP_AVAILABLE:
            reduced_embeddings_10d = self.reduce_dimensions(
                method='umap',
                n_components=10,
                use_pca=use_pca,
                umap_neighbors=max(2, min(30, len(self.embeddings) - 1)),
                umap_min_dist=0.0
            )

        reduced_embeddings = self.reduce_dimensions(
            method=dimension_reduction,
            use_pca=use_pca,
            umap_neighbors=max(2, min(30, len(self.embeddings) - 1)) if dimension_reduction == 'umap' else None,
            umap_min_dist=0.0 if dimension_reduction == 'umap' else 0.1
        )
        
        # 4. クラスタ数最適化（必要に応じて）
        optimization_results = None
        if auto_optimize_clusters and clustering_method == 'kmeans':
            # クラスタリング用の埋め込みを決定
            clustering_input = reduced_embeddings_10d if reduced_embeddings_10d is not None else self.embeddings
            
            if cluster_optimization_method == 'combined':
                optimal_k, optimization_results = self.find_optimal_clusters_combined(clustering_input)
            elif cluster_optimization_method == 'elbow':
                optimal_k, optimization_results = self.find_optimal_clusters_elbow(clustering_input)
            elif cluster_optimization_method == 'silhouette':
                optimal_k, optimization_results = self.find_optimal_clusters_silhouette(clustering_input)
            else:
                logger.warning(f"未知の最適化手法: {cluster_optimization_method}. 統合手法を使用します")
                optimal_k, optimization_results = self.find_optimal_clusters_combined(clustering_input)
            
            if optimal_k is not None:
                n_clusters = optimal_k
                logger.info(f"自動最適化により決定されたクラスタ数: {n_clusters}")
            else:
                n_clusters = 5  # デフォルト値
                logger.warning("クラスタ数自動最適化に失敗。デフォルト値5を使用します")
        elif n_clusters is None:
            n_clusters = 5  # デフォルト値
            logger.info(f"デフォルトクラスタ数を使用: {n_clusters}")
        
        # 5. クラスタリング
        # クラスタリング（10次元UMAPがあればそれを使用、なければ元の埋め込み）
        clustering_input = reduced_embeddings_10d if reduced_embeddings_10d is not None else self.embeddings
        self.perform_clustering(
            method=clustering_method,
            n_clusters=n_clusters,
            embeddings=clustering_input
        )
        
        # 6. 可視化
        self.create_visualizations(reduced_embeddings)
        
        # 7. クラスタ最適化結果の可視化
        if optimization_results is not None:
            output_dir = Path("narrative_clustering_results")
            output_dir.mkdir(exist_ok=True)
            self.plot_cluster_optimization(optimization_results, output_dir)
        
        # 8. 結果保存
        self.save_results(reduced_embeddings)
        
        # 結果表示
        logger.info("=== クラスタリング分析結果 ===")
        if self.cluster_labels is not None:
            unique_clusters = set(self.cluster_labels)
            n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
            logger.info(f"発見されたクラスタ数: {n_clusters}")
            
            for cluster_id in sorted(unique_clusters):
                if cluster_id == -1:
                    size = np.sum(self.cluster_labels == cluster_id)
                    logger.info(f"ノイズクラスタ: {size}件")
                else:
                    size = np.sum(self.cluster_labels == cluster_id)
                    logger.info(f"クラスタ {cluster_id}: {size}件")
        
        logger.info("=== 分析完了 ===")

def main():
    """メイン関数"""
    if not SKLEARN_AVAILABLE:
        print("エラー: scikit-learnが必要です。")
        return
    
    if not TRANSFORMERS_AVAILABLE:
        print("エラー: transformersが必要です。")
        return
    
    print("ゲームマニュアル物語記述クラスタリング分析を開始します...")
    
    # 分析実行
    analyzer = GameNarrativeClustering(
        bert_model_name="cl-tohoku/bert-base-japanese",
        use_sentence_transformer=True
    )
    
    analyzer.run_analysis(
        clustering_method='kmeans',
        n_clusters=None,  # 自動最適化を使用
        auto_optimize_clusters=True,
        cluster_optimization_method='combined',
        dimension_reduction='umap',
        use_pca=True
    )
    
    print("分析が完了しました！")
    print("結果ファイル:")
    print("- narrative_clustering_results/clustering_results.json")
    print("- narrative_clustering_results/cluster_documents.json")
    print("- narrative_clustering_results/clustering_results.csv")
    print("可視化ファイル:")
    print("- narrative_clustering_results/cluster_distribution.png")
    print("- narrative_clustering_results/cluster_keywords.png")
    print("- narrative_clustering_results/game_cluster_distribution.png")
    print("- narrative_clustering_results/cluster_statistics.png")
    print("クラスタ最適化結果:")
    print("- narrative_clustering_results/combined_optimization.png (または elbow_curve.png, silhouette_analysis.png)")

if __name__ == "__main__":
    main()
