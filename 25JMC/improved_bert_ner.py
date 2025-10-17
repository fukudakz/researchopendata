#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改善されたBERTベースの固有表現抽出(NER)スクリプト

cl-tohoku/bert-base-japanese-whole-word-maskingを使用した
BIOスキーマでのファインチューニング対応NERシステムです。
アノテーション済みエンティティデータを使用した強化版。
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import defaultdict, Counter
import re
from typing import Dict, List, Tuple, Set, Optional
import warnings
import torch
import os
warnings.filterwarnings('ignore')

# 日本語フォント設定
try:
    import japanize_matplotlib
    plt.rcParams['axes.unicode_minus'] = False
except ImportError:
    print("japanize_matplotlibが利用できません。日本語表示に制限があります。")

# Transformers関連のインポート
try:
    from transformers import (
        AutoTokenizer, AutoModelForTokenClassification, 
        TrainingArguments, Trainer, DataCollatorForTokenClassification
    )
    from datasets import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Transformersライブラリが利用できません。パターンベースの抽出のみ使用可能です。")
    TRANSFORMERS_AVAILABLE = False

class ImprovedBERTNERExtractor:
    """改善されたBERTベースの固有表現抽出クラス（アノテーションデータ対応）"""
    
    def __init__(self, model_name: str = "studio-ousia/luke-japanese-base-lite"):
        """初期化"""
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
        # エンティティタイプの正規化マッピング（大文字小文字を統一）
        self.entity_type_normalization = {
            'character': 'CHARACTER',
            'CHARACTER': 'CHARACTER',
            'location': 'LOCATION', 
            'LOCATION': 'LOCATION',
            'game_title': 'GAME_TITLE',
            'GAME_TITLE': 'GAME_TITLE',
            'item': 'ITEM',
            'ITEM': 'ITEM',
            'company': 'COMPANY',
            'COMPANY': 'COMPANY',
            'enemy': 'ENEMY',
            'ENEMY': 'ENEMY'
        }
        
        # 正規化されたエンティティタイプでラベルマッピングを作成
        self.label2id = {
            'O': 0,
            'B-GAME_TITLE': 1, 'I-GAME_TITLE': 2,
            'B-CHARACTER': 3, 'I-CHARACTER': 4,
            'B-LOCATION': 5, 'I-LOCATION': 6,
            'B-ENEMY': 7, 'I-ENEMY': 8,
            'B-ITEM': 9, 'I-ITEM': 10,
            'B-COMPANY': 11, 'I-COMPANY': 12
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        # ゲーム固有のエンティティ知識ベース
        self.game_entities = self._load_game_knowledge()
        
        # ストップワード
        self.stop_words = self._load_stop_words()
        
        # アノテーションデータ
        self.annotated_data = None
        
        # BERTモデルの初期化
        if TRANSFORMERS_AVAILABLE:
            self._initialize_bert()
    
    def normalize_entity_type(self, entity_type: str) -> str:
        """エンティティタイプを正規化（大文字小文字を統一）"""
        return self.entity_type_normalization.get(entity_type.lower(), entity_type.upper())
    
    def _initialize_bert(self):
        """BERTモデルとトークナイザーを初期化"""
        try:
            print(f"モデル {self.model_name} を読み込み中...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.label2id),
                id2label=self.id2label,
                label2id=self.label2id
            )
            print("モデルの読み込みが完了しました")
        except Exception as e:
            print(f"モデルの読み込みに失敗しました: {e}")
            self.model = None
            self.tokenizer = None
    
    def load_annotated_data(self, csv_path: str = 'source/annotated_entities.csv'):
        """アノテーション済みエンティティデータを読み込み"""
        try:
            print(f"アノテーションデータを読み込み中: {csv_path}")
            df = pd.read_csv(csv_path)
            print(f"CSVファイル読み込み完了: {len(df)} 行")
            
            # データの前処理
            self.annotated_data = []
            skipped_rows = 0
            
            for idx, row in df.iterrows():
                if pd.notna(row['entity_text']) and pd.notna(row['label_type']) and pd.notna(row['full_text']):
                    # エンティティタイプを正規化
                    normalized_label_type = self.normalize_entity_type(str(row['label_type']).strip())
                    self.annotated_data.append({
                        'entity_text': str(row['entity_text']).strip(),
                        'label_type': normalized_label_type,
                        'full_text': str(row['full_text']).strip(),
                        'start': int(row['start']) if pd.notna(row['start']) else 0,
                        'end': int(row['end']) if pd.notna(row['end']) else 0,
                        'game_id': str(row['game_id']) if pd.notna(row['game_id']) else '',
                        'section_type': str(row['section_type']) if pd.notna(row['section_type']) else ''
                    })
                else:
                    skipped_rows += 1
            
            print(f"アノテーションデータを読み込みました: {len(self.annotated_data)} 件")
            print(f"スキップした行数: {skipped_rows} 件")
            
            # エンティティタイプの統計（詳細）
            label_counts = Counter([item['label_type'] for item in self.annotated_data])
            print("\n=== 注釈データ統計 ===")
            print("エンティティタイプ別件数:")
            for label, count in label_counts.most_common():
                print(f"  {label}: {count}件")
            
            # ゲーム別統計
            game_counts = Counter([item['game_id'] for item in self.annotated_data])
            print(f"\nゲーム別注釈数（上位5件）:")
            for game_id, count in game_counts.most_common(5):
                print(f"  {game_id}: {count}件")
            
            # エンティティ長統計
            entity_lengths = [len(item['entity_text']) for item in self.annotated_data]
            if entity_lengths:
                print(f"\nエンティティ長統計:")
                print(f"  平均: {np.mean(entity_lengths):.1f}文字")
                print(f"  最小: {min(entity_lengths)}文字")
                print(f"  最大: {max(entity_lengths)}文字")
                
        except Exception as e:
            print(f"アノテーションデータの読み込みに失敗しました: {e}")
            import traceback
            traceback.print_exc()
            self.annotated_data = []
    
    def create_training_data_from_annotations(self) -> List[Dict]:
        """アノテーションデータからトレーニングデータを作成"""
        if not self.annotated_data:
            print("アノテーションデータが読み込まれていません")
            return []
        
        if not self.tokenizer:
            print("トークナイザーが初期化されていません")
            return []
        
        print("アノテーションデータからトレーニングデータを作成中...")
        training_data = []
        processed_texts = 0
        successful_annotations = 0
        failed_annotations = 0
        
        # テキストごとにグループ化
        text_groups = defaultdict(list)
        for item in self.annotated_data:
            text_groups[item['full_text']].append(item)
        
        print(f"処理対象テキスト数: {len(text_groups)}")
        
        for text, annotations in text_groups.items():
            processed_texts += 1
            
            # テキストをトークン化（シンプルな方法）
            tokens = self.tokenizer.tokenize(text)
            
            # ラベルを初期化（すべてO）
            labels = ['O'] * len(tokens)
            
            # 各アノテーションを処理
            for annotation in annotations:
                entity_text = annotation['entity_text']
                label_type = annotation['label_type']
                start_pos = annotation['start']
                end_pos = annotation['end']
                
                # エンティティの位置を検証
                if start_pos < len(text) and end_pos <= len(text):
                    entity_in_text = text[start_pos:end_pos]
                    
                    # エンティティテキストが一致するか確認
                    if entity_in_text == entity_text:
                        # エンティティのトークン化
                        entity_tokens = self.tokenizer.tokenize(entity_text)
                        
                        if len(entity_tokens) > 0:
                            # テキスト内でエンティティの位置を特定（文字ベース → トークンベース）
                            text_before_entity = text[:start_pos]
                            tokens_before_entity = self.tokenizer.tokenize(text_before_entity)
                            entity_start_token_idx = len(tokens_before_entity)
                            
                            # ラベルを設定（正規化されたエンティティタイプを使用）
                            if entity_start_token_idx < len(labels):
                                labels[entity_start_token_idx] = f'B-{label_type}'
                                
                                for i in range(1, len(entity_tokens)):
                                    if entity_start_token_idx + i < len(labels):
                                        labels[entity_start_token_idx + i] = f'I-{label_type}'
                                successful_annotations += 1
                            else:
                                failed_annotations += 1
                        else:
                            failed_annotations += 1
                    else:
                        failed_annotations += 1
                        if processed_texts <= 3:  # 最初の3テキストのみエラー詳細表示
                            print(f"エンティティテキスト不一致: 期待='{entity_text}', 実際='{entity_in_text}'")
                else:
                    failed_annotations += 1
                    if processed_texts <= 3:  # 最初の3テキストのみエラー詳細表示
                        print(f"位置範囲外: start={start_pos}, end={end_pos}, text_len={len(text)}")
            
            # ラベルIDに変換
            label_ids = [self.label2id.get(label, 0) for label in labels]
            
            # 不正なラベルをチェック
            unknown_labels = [label for label in labels if label not in self.label2id]
            if unknown_labels:
                print(f"未知のラベル: {set(unknown_labels)}")
            
            training_data.append({
                'tokens': tokens,
                'labels': label_ids,
                'text': text
            })
        
        print(f"\n=== トレーニングデータ作成結果 ===")
        print(f"処理テキスト数: {processed_texts}")
        print(f"成功した注釈数: {successful_annotations}")
        print(f"失敗した注釈数: {failed_annotations}")
        print(f"成功率: {successful_annotations/(successful_annotations+failed_annotations)*100:.1f}%")
        print(f"作成されたトレーニングデータ数: {len(training_data)}")
        
        # ラベル分布の確認
        all_labels = []
        for item in training_data:
            all_labels.extend([self.id2label[label_id] for label_id in item['labels']])
        
        label_counts = Counter(all_labels)
        print(f"\nラベル分布:")
        for label, count in label_counts.most_common():
            print(f"  {label}: {count}")
        
        return training_data
    
    def create_training_data(self, texts: List[str]) -> List[Dict]:
        """ファインチューニング用のトレーニングデータを作成（アノテーションデータ優先）"""
        # アノテーションデータがある場合はそれを使用
        if self.annotated_data:
            return self.create_training_data_from_annotations()
        
        # フォールバック: パターンベースのトレーニングデータ作成
        training_data = []
        
        for text in texts:
            # テキストをトークン化
            tokens = self.tokenizer.tokenize(text)
            
            # ラベルを初期化（すべてO）
            labels = ['O'] * len(tokens)
            
            # ゲーム固有のエンティティを検出してラベルを設定
            for entity_type, entities in self.game_entities.items():
                for entity in entities:
                    if entity in text:
                        # エンティティの位置を検出
                        entity_tokens = self.tokenizer.tokenize(entity)
                        if len(entity_tokens) > 0:
                            # テキスト内でエンティティを検索
                            for i in range(len(tokens) - len(entity_tokens) + 1):
                                if tokens[i:i+len(entity_tokens)] == entity_tokens:
                                    # BIOスキーマでラベルを設定
                                    labels[i] = f'B-{entity_type}'
                                    for j in range(1, len(entity_tokens)):
                                        if i + j < len(labels):
                                            labels[i + j] = f'I-{entity_type}'
                                    
                                    # 重複を避けるため、処理済みのトークンをスキップ
                                    break
            
            # ラベルIDに変換
            label_ids = [self.label2id.get(label, 0) for label in labels]
            
            training_data.append({
                'tokens': tokens,
                'labels': label_ids,
                'text': text
            })
        
        return training_data
    
    def _load_game_knowledge(self) -> Dict[str, List[str]]:
        """ゲーム固有の知識ベースを読み込み"""
        return {
            'character': [
                # マリオシリーズ
                'マリオ', 'ルイージ', 'ピーチ姫', 'キノピオ', 'クッパ', 'ヨッシー',
                'ワリオ', 'ワルイージ', 'ドンキーコング', 'ディディーコング',
                'キングクルール', 'キングデデデ', 'メタナイト', 'カービー',
                # ゼルダシリーズ
                'リンク', 'ゼルダ姫', 'ガノンドルフ', 'イムパ', 'ナビ',
                'ミドナ', 'フィ', 'ダルニア', 'ローブ', 'シーク',
                # その他のゲーム
                'サムス', 'アラン', 'ドラキュラ', 'シモン', 'ベルモンド',
                'ソニック', 'メガマン', 'ロックマン', 'ビル', 'ランス'
            ],
            'location': [
                # マリオシリーズ
                'キノコ王国', 'キノコワールド', 'ドンキーコングアイランド',
                'ドリームランド', 'ポップスター', 'ハルカンドラ',
                # ゼルダシリーズ
                'ハイラル', 'カカリコ村', 'ハイラル城', 'デスマウンテン',
                'ゾーラの里', 'ゴロンシティ', 'ゲルドの谷',
                # その他のゲーム
                'トランシルバニア', 'グラディウス', 'アトランチス',
                'プププランド', 'ライラック', 'ウル'
            ],
            'item': [
                # マリオシリーズ
                'スーパーキノコ', 'ファイアフラワー', 'スター', '1UPキノコ',
                'ハンマー', 'タナカン', 'スーパーリーフ',
                # ゼルダシリーズ
                'マスターソード', 'トライフォース', 'ルピー', 'ハート',
                '魔法のメダル', 'ボム', '弓矢',
                # その他のゲーム
                'メトロイド', 'スターロッド', 'ソロモンの鍵'
            ],
            'game_title': [
                'スーパーマリオブラザーズ', 'スーパーマリオブラザーズ3',
                'スーパーマリオUSA', 'マリオオープンゴルフ',
                'ゼルダの伝説', 'リンクの冒険',
                'ポケットモンスター', 'ドラゴンクエスト',
                'ファイナルファンタジー', 'メタルギア',
                'バイオハザード', 'モンスターハンター'
            ],
            'company': [
                '任天堂', 'Nintendo', 'ソニー', 'Sony', 'マイクロソフト', 'Microsoft',
                'スクウェア・エニックス', 'Square Enix', 'カプコン', 'Capcom',
                'コナミ', 'Konami', 'セガ', 'SEGA', 'バンダイナムコ', 'Bandai Namco'
            ],
            'enemy': [
                'クリーチャー', 'モンスター', 'ボス', '敵', '悪魔', 'ドラゴン',
                'ゴブリン', 'オーク', 'トロール', 'スケルトン', 'ゾンビ'
            ]
        }
    
    def _load_stop_words(self) -> Set[str]:
        """ストップワードを読み込み"""
        return {
            'の', 'に', 'は', 'を', 'が', 'と', 'で', 'から', 'まで', 'より',
            'や', 'か', 'も', 'に', 'へ', 'で', 'を', 'が', 'は', 'の', 'に',
            'て', 'で', 'と', 'から', 'まで', 'より', 'や', 'か', 'も',
            'ある', 'いる', 'なる', 'する', 'できる', 'れる', 'られる',
            'です', 'ます', 'でした', 'ました', 'です', 'ます',
            'この', 'その', 'あの', 'どの', 'これ', 'それ', 'あれ', 'どれ',
            'ここ', 'そこ', 'あそこ', 'どこ', 'こちら', 'そちら', 'あちら', 'どちら',
            '今', '昨日', '今日', '明日', '去年', '今年', '来年',
            '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
            '百', '千', '万', '億', '兆'
        }
    
    def train_model(self, training_data: List[Dict], epochs: int = 3):
        """モデルをファインチューニング"""
        if not self.model or not self.tokenizer:
            print("モデルが初期化されていません")
            return
        
        try:
            print("ファインチューニングを開始します...")
            
            # トレーニングデータを適切な形式に変換
            processed_data = []
            for item in training_data:
                # テキストをトークン化してinput_idsを取得
                encoding = self.tokenizer(
                    item['text'],
                    truncation=True,
                    max_length=512,
                    padding='max_length',
                    return_tensors='pt'
                )
                
                # ラベルを適切な長さに調整
                labels = item['labels']
                if len(labels) > 512:
                    labels = labels[:512]
                elif len(labels) < 512:
                    labels = labels + [0] * (512 - len(labels))
                
                processed_data.append({
                    'input_ids': encoding['input_ids'].squeeze().tolist(),
                    'attention_mask': encoding['attention_mask'].squeeze().tolist(),
                    'labels': labels
                })
            
            # データセットを作成
            dataset = Dataset.from_list(processed_data)
            
            # データコレーター
            data_collator = DataCollatorForTokenClassification(
                tokenizer=self.tokenizer,
                return_tensors="pt"
            )
            
            # トレーニング引数（軽めの設定）
            training_args = TrainingArguments(
                output_dir="./bert_ner_model",
                num_train_epochs=min(epochs, 2),  # エポック数を制限
                per_device_train_batch_size=4,  # バッチサイズを小さく
                per_device_eval_batch_size=4,
                learning_rate=2e-5,  # 学習率を少し上げる
                warmup_steps=50,  # ウォームアップステップをさらに減らす
                weight_decay=0.01,
                logging_dir="./logs",
                logging_steps=10,
                save_steps=1000,
                evaluation_strategy="no",  # 評価データセットがないため無効化
                save_total_limit=2,
                no_cuda=True,  # GPUを使用しない（MPS問題を回避）
            )
            
            # トレーナー
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
            )
            
            # トレーニング実行
            trainer.train()
            
            print("ファインチューニングが完了しました")
            
        except Exception as e:
            print(f"トレーニングエラー: {e}")
    
    def extract_entities_bert(self, text: str, confidence_threshold: float = 0.3, debug: bool = False) -> Dict[str, List[Dict]]:
        """LUKEを使用した固有表現抽出（BIOスキーマ対応）"""
        if not self.model or not self.tokenizer:
            return {'entities': []}
        
        try:
            # テキストをトークン化
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            
            # CPUに移動（MPS問題を回避）
            inputs = {k: v.cpu() for k, v in inputs.items()}
            self.model = self.model.cpu()
            
            # 予測実行
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=2)
                predictions = torch.argmax(logits, dim=2)
            
            # トークンとラベルを取得
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())
            labels = [self.id2label[pred.item()] for pred in predictions.squeeze()]
            
            # 確信度を取得
            probs = probabilities.squeeze()
            confidences = [torch.max(probs[i]).item() for i in range(len(tokens))]
            
            if debug:
                print(f"\n=== LUKE抽出デバッグ情報 ===")
                print(f"入力テキスト: {text[:100]}...")
                print(f"トークン数: {len(tokens)}")
                non_o_labels = [(i, token, label, conf) for i, (token, label, conf) in enumerate(zip(tokens, labels, confidences)) if label != 'O']
                print(f"非Oラベルの予測数: {len(non_o_labels)}")
                
                # 閾値を超える予測の数
                above_threshold = [item for item in non_o_labels if item[3] >= confidence_threshold]
                print(f"閾値{confidence_threshold}以上の予測数: {len(above_threshold)}")
                
                # 全ての非O予測を表示（閾値以下も含む）
                for i, token, label, conf in non_o_labels[:15]:  # 上位15件表示
                    status = "✓" if conf >= confidence_threshold else "✗"
                    print(f"  {status} {i}: '{token}' -> {label} (conf: {conf:.3f})")
                
                # 最高確信度の予測
                if non_o_labels:
                    max_conf_item = max(non_o_labels, key=lambda x: x[3])
                    print(f"最高確信度: '{max_conf_item[1]}' -> {max_conf_item[2]} (conf: {max_conf_item[3]:.3f})")
            
            # エンティティを抽出（確信度閾値を適用）
            entities = []
            current_entity = None
            
            for i, (token, label, confidence) in enumerate(zip(tokens, labels, confidences)):
                # 確信度が閾値を下回る場合はOとして扱う
                if confidence < confidence_threshold:
                    label = 'O'
                
                if label.startswith('B-'):
                    # 新しいエンティティの開始
                    if current_entity:
                        entities.append(current_entity)
                    
                    entity_type = label[2:]  # B-CHARACTER -> CHARACTER
                    # エンティティタイプを正規化
                    normalized_entity_type = self.normalize_entity_type(entity_type)
                    # サブワードプレフィックスを除去
                    clean_token = token.replace('##', '')
                    current_entity = {
                        'text': clean_token,
                        'type': normalized_entity_type,
                        'start': i,
                        'end': i,
                        'score': confidence
                    }
                
                elif label.startswith('I-') and current_entity:
                    # エンティティの継続
                    entity_type = label[2:]
                    # エンティティタイプを正規化
                    normalized_entity_type = self.normalize_entity_type(entity_type)
                    if normalized_entity_type == current_entity['type']:
                        # サブワードプレフィックスを除去してから結合
                        clean_token = token.replace('##', '')
                        current_entity['text'] += clean_token
                        current_entity['end'] = i
                        # 確信度の平均を取る
                        current_entity['score'] = (current_entity['score'] + confidence) / 2
                
                elif label == 'O':
                    # エンティティの終了
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None
            
            # 最後のエンティティを追加
            if current_entity:
                entities.append(current_entity)
            
            # エンティティをクリーンアップ
            cleaned_entities = []
            for entity in entities:
                clean_text = self._clean_entity_text(entity['text'])
                if clean_text and self._is_valid_entity(clean_text, entity.get('type')):
                    entity['text'] = clean_text
                    cleaned_entities.append(entity)
            
            if debug and cleaned_entities:
                print(f"抽出されたエンティティ数: {len(cleaned_entities)}")
                for entity in cleaned_entities[:5]:  # 上位5件表示
                    print(f"  '{entity['text']}' ({entity['type']}, score: {entity['score']:.3f})")
            
            return {'entities': cleaned_entities}
            
        except Exception as e:
            print(f"LUKE抽出エラー: {e}")
            import traceback
            traceback.print_exc()
            return {'entities': []}
    
    def _clean_entity_text(self, text: str) -> str:
        """エンティティテキストをクリーンアップ"""
        # 特殊トークンを除去
        text = re.sub(r'\[UNK\]|\[CLS\]|\[SEP\]', '', text)
        
        # サブワードプレフィックスを除去（念のため）
        text = re.sub(r'##', '', text)
        
        # 空白を除去
        text = text.strip()
        
        return text
    
    def extract_entities_pattern(self, text: str) -> Dict[str, List[str]]:
        """パターンベースの固有表現抽出（改善版）"""
        entities = defaultdict(list)
        
        # ゲーム固有の用語を検索（エンティティタイプを正規化）
        for category, terms in self.game_entities.items():
            normalized_category = self.normalize_entity_type(category)
            for term in terms:
                if term in text:
                    entities[normalized_category].append(term)
        
        # 日本語固有表現パターン（改善版 - より具体的で正確なパターン）
        patterns = {
            'CHARACTER': [
                # 敬語付き人名（より具体的）
                r'([一-龯]{2,4}さん)', r'([一-龯]{2,4}君)', r'([一-龯]{2,4}様)',
                r'([一-龯]{2,4}姫)', r'([一-龯]{2,4}王)', r'([一-龯]{2,4}王子)',
                r'([一-龯]{2,4}くん)', r'([一-龯]{2,4}ちゃん)',
                # 中黒で区切られたカタカナ名（より人名らしい）
                r'([ァ-ヶ]{2,4}・[ァ-ヶ]{2,4})',
                # 特定のゲームキャラクター名パターン
                r'([ァ-ヶ]{2,4}ー[ァ-ヶ]{2,4})',  # ハイフン区切り
                # 汎用的なカタカナ抽出はコメントアウト（誤分類を防ぐため）
                # r'([ァ-ヶ]{2,6})',  # カタカナ人名 - 汎用性が高すぎるため無効化
            ],
            'LOCATION': [
                r'([一-龯]{2,4}国)', r'([一-龯]{2,4}王国)', r'([一-龯]{2,4}城)',
                r'([一-龯]{2,4}村)', r'([一-龯]{2,4}町)', r'([一-龯]{2,4}市)',
                r'([一-龯]{2,4}山)', r'([一-龯]{2,4}川)', r'([一-龯]{2,4}湖)',
                r'([一-龯]{2,4}島)', r'([一-龯]{2,4}大陸)', r'([一-龯]{2,4}地方)',
                # より具体的な地名パターン
                r'([ァ-ヶ]{2,4}王国)', r'([ァ-ヶ]{2,4}ワールド)', r'([ァ-ヶ]{2,4}ランド)',
            ],
            'ITEM': [
                r'([一-龯]{2,4}の杖)', r'([一-龯]{2,4}の剣)', r'([一-龯]{2,4}の盾)',
                r'([一-龯]{2,4}の鍵)', r'([一-龯]{2,4}の本)', r'([一-龯]{2,4}の石)',
                r'([一-龯]{2,4}の薬)', r'([一-龯]{2,4}の呪文)', r'([一-龯]{2,4}の魔法)',
                # より具体的なアイテムパターン
                r'([ァ-ヶ]{2,4}キノコ)', r'([ァ-龯]{2,4}フラワー)', r'([ァ-龯]{2,4}スター)',
                r'([ァ-龯]{2,4}ソード)', r'([ァ-龯]{2,4}シールド)', r'([ァ-龯]{2,4}リング)',
            ]
        }
        
        for entity_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, text)
                for match in matches:
                    if len(match) >= 2 and match not in self.stop_words:
                        # より厳密なフィルタリング
                        if self._is_valid_pattern_entity(match, entity_type):
                            entities[entity_type].append(match)
        
        return dict(entities)
    
    def extract_entities_combined(self, text: str, debug: bool = False) -> Dict:
        """LUKEとパターンを組み合わせた抽出（BIOスキーマ対応）"""
        results = {
            'bert_entities': [],
            'pattern_entities': {},
            'combined_entities': defaultdict(list)
        }
        
        # LUKEによる抽出（厳格な閾値と厳格なホワイトリスト）
        bert_results = self.extract_entities_bert(text, confidence_threshold=0.1, debug=debug)
        if debug and bert_results['entities']:
            print(f"🎯 LUKE実予測評価: {len(bert_results['entities'])}件（閾値=0.1, 最小限フィルタ）")
        results['bert_entities'] = bert_results['entities']
        
        # パターンベースによる抽出
        pattern_results = self.extract_entities_pattern(text)
        results['pattern_entities'] = pattern_results
        
        # 統合とフィルタリング
        all_entities = set()
        
        # BERTエンティティを統合（対象エンティティタイプのみ）
        valid_entity_types = {'CHARACTER', 'LOCATION', 'ITEM', 'ENEMY', 'GAME_TITLE'}
        for entity in bert_results['entities']:
            entity_text = entity['text']
            entity_type = entity['type']
            # 対象エンティティタイプのみ処理
            if entity_type in valid_entity_types and self._is_valid_entity(entity_text, entity_type):
                all_entities.add(entity_text)
                results['combined_entities'][entity_type].append(entity_text)
        
        # パターンエンティティを統合（エンティティタイプを正規化）
        for entity_type, entities in pattern_results.items():
            normalized_entity_type = self.normalize_entity_type(entity_type)
            for entity in entities:
                if self._is_valid_entity(entity, normalized_entity_type):
                    all_entities.add(entity)
                    results['combined_entities'][normalized_entity_type].append(entity)
        
        return results
    
    def _is_valid_entity(self, entity: str, entity_type: str = None) -> bool:
        """エンティティが有効かどうかをチェック（厳格版）"""
        if not entity or len(entity) < 2:
            return False
        
        # ストップワードをチェック
        if entity in self.stop_words:
            return False
        
        # 数字のみの場合は除外
        if entity.isdigit():
            return False
        
        # 特殊文字のみの場合は除外
        if re.match(r'^[^\w\s]+$', entity):
            return False
        
        # 長すぎる場合は除外
        if len(entity) > 30:
            return False
        
        # 一般的な動詞・助動詞・形容詞を除外（厳格）
        common_verbs_adjectives = {
            'である', 'いた', 'した', 'なった', 'ある', 'いる', 'する', 'なる',
            'だった', 'でした', 'ました', 'です', 'ます', 'だ', 'である',
            'のだ', 'のです', 'という', 'として', 'について', 'により',
            '住む', '来た', '行く', '見る', '聞く', '話す', '思う', '考える',
            'がある', 'がいる', 'ができる', 'をする', 'になる', 'となる',
            'といった', 'による', 'での', 'への', 'からの', 'までの',
            'った', 'ると', 'れば', 'ても', 'ては', 'では', 'には',
            'だけ', 'まで', 'から', 'より', 'ほど', 'など', 'なら',
            'こと', 'もの', 'ところ', 'とき', 'ため', 'よう', 'ほう',
            '中に', '上に', '下に', '前に', '後に', '横の', '近くの',
            'していた', 'している', 'されている', 'となって', 'になって',
            'ていた', 'ている', 'てる', 'たが', 'だが', 'しかし',
            'そして', 'また', 'さらに', 'ついに', 'やがて', 'すると',
            'それで', 'そこで', 'ところで', 'ただし', 'しかも',
            'あなた', 'みんな', 'みなさん', 'きみ', '君', '僕', '私',
            'テレビ', 'ゲーム', 'プレイ', 'プレイヤー', 'コントローラー',
            'ボタン', 'スタート', 'セレクト', 'ポーズ', 'リセット',
            # ENEMY/ITEMで特に問題の多い語彙を追加
            'して', 'してくれる', 'ける', '訪れる', '合わせる', 'える', 'さい',
            'りました', 'ください', 'であり', 'たち', 'のである', 'なのだ',
            '持ち', 'の末', '連れ', '次の', '問い', 'を解', 'を倒', 'げん',
            'によって', 'になっちゃう', '待って', '引き', 'るとき', 'でお',
            '力で', 'のため', 'を持ち', '使い', '探し', 'がやって', 'と呼ばれ',
            '者に', '率いる', 'を終え', 'がんば', '時間', '私は', '者の',
            '形見', '形の', 'た異', 'ティ', 'ノン', 'シル', 'ベル', 'ハル',
            'モンの', 'ーク', 'ザー', '前作', '遊び', '紹介', 'コク', 'コン',
            '気をつけて', 'の奥', '全部で', '怒鳴', 'にお答え', 'の世界',
            '目的地', 'がこの', 'ふた', 'につく', 'ひとつ', 'の先', 'ひとりで',
            '勇気', '連邦', '連邦警察', '連邦警察は', '金の', 'ウォ',
            # さらに追加のフィルタリング
            '惑星', 'コース', '規定', '調査', 'しかいない', '語った', 'やってきた',
            '英雄', '青年', 'エネルギー', '完結', 'に参加', 'ソード', '生涯',
            'られている', 'しましょう', '考え', 'ールド', 'できなかった', 'ってお',
            '守護', 'とても', '語り', '復興', 'サム', '増殖', '破壊', '照射',
            '頑張ってください', 'コースの', 'マップ', '世界に', '経過', '成果',
            'きた', '魔法', '研究', '発見', '道士', '文字', '文書', '知識',
            '技術', '科学', '学習', '訓練', '練習', '実験', '試験', '検査',
            'できる', 'できた', 'できない', 'かった', 'ました', 'ません',
            'ような', 'みたい', 'らしい', 'だろう', 'でしょう', 'かもしれない'
        }
        
        if entity in common_verbs_adjectives:
            return False
        
        # 単一ひらがな・カタカナの場合は除外
        if len(entity) == 1 and (
            '\u3042' <= entity <= '\u3096' or  # ひらがな
            '\u30a2' <= entity <= '\u30f6'     # カタカナ
        ):
            return False
        
        # 助詞・接続詞パターンを除外
        particle_patterns = [
            r'^[はをにがのでとからまでより]+$',  # 助詞のみ
            r'^[っとてでだじゃたやよね]+$',      # 語尾のみ
            r'^[▁]+',                          # 不適切なプレフィックス
            r'^[、。！？]+$',                   # 句読点のみ
            r'^<s>$',                          # 開始マーカー
            r'^</s>$',                         # 終了マーカー
            r'^[▁「」『』（）()]+$',           # 括弧・引用符のみ
            r'^じゃあ$',                       # 接続詞
            r'^だから$',                       # 接続詞
            r'^それで$',                       # 接続詞
            r'^でも$',                         # 接続詞
            r'^しかし$',                       # 接続詞
        ]
        
        for pattern in particle_patterns:
            if re.match(pattern, entity):
                return False
        
        # エンティティタイプ別の追加検証
        return self._validate_entity_by_type(entity, entity_type)
    
    def _validate_entity_by_type(self, entity: str, entity_type: str = None) -> bool:
        """エンティティタイプ別の詳細検証（LUKE予測評価版）"""
        if not entity_type:
            return True
        
        # 明らかに不正な語彙のみ除外（LUKEの予測能力を評価するため）
        obvious_invalid_words = {
            # 助詞・助動詞・接続詞
            'の', 'が', 'を', 'に', 'で', 'と', 'は', 'も', 'から', 'まで',
            'である', 'です', 'ます', 'だった', 'でした', 'ました',
            'そして', 'また', 'しかし', 'でも', 'ところで', 'すると',
            # 特殊文字・記号
            '<s>', '</s>', '、', '。', '！', '？', '「', '」',
            # 明らかな動詞・形容詞の活用形
            'した', 'して', 'される', 'なった', 'になる', 'による',
            'だろう', 'でしょう', 'かもしれない', 'らしい'
        }
        
        if entity in obvious_invalid_words:
            return False
        
        # 1文字のエンティティは除外
        if len(entity) < 2:
            return False
        
        # 数字のみのエンティティは除外
        if entity.isdigit():
            return False
        
        # それ以外はLUKEの予測を信頼
        return True
    
    def _is_valid_pattern_entity(self, entity: str, entity_type: str) -> bool:
        """パターンベース抽出で得られたエンティティが有効かどうかをチェック"""
        if not entity or len(entity) < 2:
            return False
        
        # ストップワードをチェック
        if entity in self.stop_words:
            return False
        
        # 数字のみの場合は除外
        if entity.isdigit():
            return False
        
        # 特殊文字のみの場合は除外
        if re.match(r'^[^\w\s]+$', entity):
            return False
        
        # 長すぎる場合は除外（30文字以上）
        if len(entity) > 30:
            return False
        
        # エンティティタイプ別の追加フィルタリング
        if entity_type == 'CHARACTER':
            # キャラクター名として不適切な一般的なゲーム用語を除外
            game_terms = {
                'テレビ', 'アクション', 'クリア', 'コンティニュー', 'ゲーム', 'プレイ',
                'ステージ', 'レベル', 'スコア', 'ポイント', 'ライフ', 'エナジー',
                'パワー', 'スピード', 'ジャンプ', 'アタック', 'ディフェンス',
                'アイテム', 'コイン', 'ボーナス', 'エクストラ', 'スペシャル',
                'ノーマル', 'ハード', 'イージー', 'ミディアム', 'ベスト', 'ワースト'
            }
            if entity in game_terms:
                return False
            
            # 一般的な動詞・形容詞を除外
            common_words = {
                'スタート', 'ストップ', 'セーブ', 'ロード', 'リセット', 'パス',
                'セレクト', 'キャンセル', 'オプション', 'メニュー', 'タイトル',
                'エンディング', 'オープニング', 'デモ', 'ムービー', 'サウンド',
                'ボリューム', 'スピーカー', 'マイク', 'コントローラー', 'ボタン'
            }
            if entity in common_words:
                return False
        
        elif entity_type == 'LOCATION':
            # 地名として不適切な一般的な用語を除外
            location_stop_words = {
                'エリア', 'ゾーン', 'フィールド', 'マップ', 'ワールド', 'ステージ',
                'レベル', 'フロア', 'ルーム', 'チャンネル', 'ライン', 'パス'
            }
            if entity in location_stop_words:
                return False
        
        elif entity_type == 'ITEM':
            # アイテムとして不適切な一般的な用語を除外
            item_stop_words = {
                'アイテム', 'アイテム', 'グッズ', 'ツール', 'ギア', 'エクイップ',
                'アクセサリー', 'オプション', 'パーツ', 'コンポーネント'
            }
            if entity in item_stop_words:
                return False
        
        return True
    
    def extract_from_manuals(self, data: Dict) -> Dict:
        """マニュアルデータから固有表現を抽出（BIOスキーマ対応）"""
        all_results = {
            'all_entities': defaultdict(list),
            'manual_results': {},
            'entity_statistics': {},
            'bert_entities': [],
            'pattern_entities': defaultdict(list)
        }
        
        for manual in data['manuals']:
            manual_title = manual['manual_title']
            game_id = manual['game_id']
            
            manual_results = {
                'title': manual_title,
                'entities': defaultdict(list),
                'bert_entities': [],
                'pattern_entities': defaultdict(list),
                'sections': []
            }
            
            for section in manual['narrative_sections']:
                section_results = {
                    'head': section.get('head', ''),
                    'entities': defaultdict(list),
                    'bert_entities': [],
                    'pattern_entities': defaultdict(list)
                }
                
                for p_idx, paragraph in enumerate(section['paragraphs']):
                    # デバッグ表示を無効化
                    debug_mode = False
                    
                    paragraph_results = self.extract_entities_combined(paragraph, debug=debug_mode)
                    
                    # LUKEエンティティ
                    section_results['bert_entities'].extend(paragraph_results['bert_entities'])
                    manual_results['bert_entities'].extend(paragraph_results['bert_entities'])
                    all_results['bert_entities'].extend(paragraph_results['bert_entities'])
                    
                    # パターンエンティティ
                    for entity_type, entities in paragraph_results['pattern_entities'].items():
                        section_results['pattern_entities'][entity_type].extend(entities)
                        manual_results['pattern_entities'][entity_type].extend(entities)
                        all_results['pattern_entities'][entity_type].extend(entities)
                    
                    # 統合エンティティ
                    for entity_type, entities in paragraph_results['combined_entities'].items():
                        section_results['entities'][entity_type].extend(entities)
                        manual_results['entities'][entity_type].extend(entities)
                        all_results['all_entities'][entity_type].extend(entities)
                
                manual_results['sections'].append(section_results)
            
            all_results['manual_results'][game_id] = manual_results
        
        # 統計情報の計算
        all_results['entity_statistics'] = self._calculate_entity_statistics(all_results['all_entities'])
        
        return all_results
    
    def _calculate_entity_statistics(self, entities: Dict[str, List[str]]) -> Dict:
        """エンティティ統計の計算"""
        stats = {}
        
        for entity_type, entity_list in entities.items():
            entity_counts = Counter(entity_list)
            stats[entity_type] = {
                'total_count': len(entity_list),
                'unique_count': len(entity_counts),
                'top_entities': entity_counts.most_common(20),
                'frequency_distribution': dict(entity_counts)
            }
        
        return stats

class ImprovedNetworkAnalyzer:
    """改善されたネットワーク分析クラス"""
    
    def __init__(self):
        """初期化"""
        self.graph = nx.Graph()
        self.entity_cooccurrence = defaultdict(int)
    
    def build_network(self, ner_results: Dict) -> Dict:
        """ネットワークを構築"""
        # エンティティ共起ネットワーク
        self._build_entity_cooccurrence_network(ner_results)
        
        return {
            'entity_network': self.graph
        }
    
    def _build_entity_cooccurrence_network(self, ner_results: Dict):
        """エンティティ共起ネットワークを構築"""
        for game_id, manual_data in ner_results['manual_results'].items():
            all_entities_in_manual = []
            for entity_type, entities in manual_data['entities'].items():
                all_entities_in_manual.extend(entities)
            
            for i, entity1 in enumerate(all_entities_in_manual):
                for entity2 in all_entities_in_manual[i+1:]:
                    if entity1 != entity2:
                        pair = tuple(sorted([entity1, entity2]))
                        self.entity_cooccurrence[pair] += 1
        
        for (entity1, entity2), weight in self.entity_cooccurrence.items():
            if weight >= 1:
                self.graph.add_edge(entity1, entity2, weight=weight, type='cooccurrence')
    
    def analyze_network(self) -> Dict:
        """ネットワーク分析を実行"""
        if len(self.graph.nodes()) == 0:
            return {}
        
        return {
            'basic_stats': {
                'nodes': self.graph.number_of_nodes(),
                'edges': self.graph.number_of_edges(),
                'density': nx.density(self.graph),
                'average_clustering': nx.average_clustering(self.graph),
                'average_shortest_path': nx.average_shortest_path_length(self.graph) if nx.is_connected(self.graph) else None
            },
            'centrality': {
                'degree': nx.degree_centrality(self.graph),
                'betweenness': nx.betweenness_centrality(self.graph),
                'closeness': nx.closeness_centrality(self.graph),
                'eigenvector': nx.eigenvector_centrality(self.graph, max_iter=1000) if len(self.graph.nodes()) > 1 else {}
            },
            'communities': {
                'connected_components': [list(component) for component in nx.connected_components(self.graph)],
                'cliques': list(nx.find_cliques(self.graph))
            }
        }

class ImprovedBERTNERAnalyzer:
    """改善されたBERTベースのNER分析のメインクラス（BIOスキーマ対応）"""
    
    def __init__(self, model_name: str = "studio-ousia/luke-japanese-base-lite"):
        """初期化"""
        self.extractor = ImprovedBERTNERExtractor(model_name)
        self.network_analyzer = ImprovedNetworkAnalyzer()
        self.results = {}
    
    def train_on_data(self, data: Dict, epochs: int = 3, use_annotated_data: bool = True):
        """データでモデルをファインチューニング"""
        print("ファインチューニング用データを準備中...")
        
        # アノテーションデータを読み込み（優先）
        if use_annotated_data:
            self.extractor.load_annotated_data()
        
        # アノテーションデータがない場合のフォールバック
        if not self.extractor.annotated_data:
            print("アノテーションデータが見つからないため、パターンベースのデータを使用します")
            # トレーニング用テキストを収集
            training_texts = []
            for manual in data['manuals']:
                for section in manual['narrative_sections']:
                    for paragraph in section['paragraphs']:
                        training_texts.append(paragraph)
            
            # トレーニングデータを作成
            training_data = self.extractor.create_training_data(training_texts)
        else:
            # アノテーションデータからトレーニングデータを作成
            training_data = self.extractor.create_training_data_from_annotations()
        
        # ファインチューニング実行
        if training_data:
            self.extractor.train_model(training_data, epochs)
        else:
            print("トレーニングデータが作成できませんでした")
    
    def analyze(self, input_file: str = 'cleaned_data/cleaned_narrative_sections.json', train_model: bool = True, use_annotated_data: bool = True):
        """メイン分析処理"""
        print("改善されたBERTベースの固有表現抽出とネットワーク分析を開始します...")
        
        # データ読み込み
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # ファインチューニング（オプション）
        if train_model:
            self.train_on_data(data, use_annotated_data=use_annotated_data)
        
        # 固有表現抽出
        print("LUKEによる固有表現を抽出中...")
        ner_results = self.extractor.extract_from_manuals(data)
        
        # ネットワーク構築
        print("ネットワークを構築中...")
        networks = self.network_analyzer.build_network(ner_results)
        
        # ネットワーク分析
        print("ネットワーク分析を実行中...")
        network_analysis = self.network_analyzer.analyze_network()
        
        # 結果を保存
        self.results = {
            'ner_results': ner_results,
            'networks': networks,
            'network_analysis': network_analysis
        }
        
        # 結果をファイルに保存
        self.save_results()
        
        # 可視化
        self.create_visualizations()
        
        print("分析完了！")
        return self.results
    
    def save_results(self):
        """結果をファイルに保存"""
        os.makedirs('improved_bert_ner_results', exist_ok=True)
        
        # 結果をJSONファイルに保存
        results_data = {
            'entity_statistics': self.results['ner_results']['entity_statistics'],
            'network_analysis': self.results['network_analysis'],
            'manual_summary': self._create_manual_summary()
        }
        
        with open('improved_bert_ner_results/improved_luke_ner_results.json', 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        # エンティティリストをCSVに保存
        self._save_entity_csv()
        
        # LUKE専用エンティティリストをCSVに保存
        self._save_luke_only_entity_csv()
        
        # 詳細な分析結果も保存
        self._save_detailed_results()
        
        print("結果を保存しました:")
        print("- improved_bert_ner_results/improved_luke_ner_results.json")
        print("- improved_bert_ner_results/improved_luke_entity_list.csv (統合)")
        print("- improved_bert_ner_results/luke_only_entity_stats.csv (LUKE専用統計)")
        print("- improved_bert_ner_results/luke_only_entities_detail.csv (LUKE専用詳細)")
        print("- improved_bert_ner_results/detailed_analysis.json")
    
    def _create_manual_summary(self) -> Dict:
        """マニュアル別サマリーを作成"""
        summary = {}
        
        for game_id, manual_data in self.results['ner_results']['manual_results'].items():
            summary[game_id] = {
                'title': manual_data['title'],
                'entity_counts': {entity_type: len(entities) for entity_type, entities in manual_data['entities'].items()},
                'bert_entity_count': len(manual_data['bert_entities']),
                'pattern_entity_count': sum(len(entities) for entities in manual_data['pattern_entities'].values()),
                'section_count': len(manual_data['sections'])
            }
        
        return summary
    
    def _save_entity_csv(self):
        """エンティティリストをCSVに保存"""
        entity_df = []
        
        for entity_type, entities in self.results['ner_results']['all_entities'].items():
            entity_counts = Counter(entities)
            for entity, count in entity_counts.items():
                entity_df.append({
                    'entity_type': entity_type,
                    'entity': entity,
                    'count': count,
                    'frequency': count / len(entities) if entities else 0
                })
        
        entity_df = pd.DataFrame(entity_df)
        entity_df.to_csv('improved_bert_ner_results/improved_luke_entity_list.csv', index=False, encoding='utf-8')
    
    def _save_luke_only_entity_csv(self):
        """LUKE抽出のみのエンティティリストをCSVに保存"""
        luke_entity_df = []
        
        # BERTエンティティのみを抽出（対象エンティティタイプのみ）
        valid_entity_types = {'CHARACTER', 'LOCATION', 'ITEM', 'ENEMY', 'GAME_TITLE'}
        for game_id, manual_data in self.results['ner_results']['manual_results'].items():
            for entity in manual_data['bert_entities']:
                if entity['type'] in valid_entity_types:
                    luke_entity_df.append({
                        'game_id': game_id,
                        'manual_title': manual_data['title'],
                        'entity_type': entity['type'],
                        'entity_text': entity['text'],
                        'confidence_score': entity.get('score', 0.0)
                    })
        
        luke_df = pd.DataFrame(luke_entity_df)
        
        # エンティティタイプ別の統計を計算
        luke_stats_df = []
        if not luke_df.empty:
            for entity_type in luke_df['entity_type'].unique():
                type_df = luke_df[luke_df['entity_type'] == entity_type]
                entity_counts = type_df['entity_text'].value_counts()
                
                for entity, count in entity_counts.items():
                    luke_stats_df.append({
                        'entity_type': entity_type,
                        'entity': entity,
                        'count': count,
                        'frequency': count / len(type_df),
                        'avg_confidence': type_df[type_df['entity_text'] == entity]['confidence_score'].mean()
                    })
        
        luke_stats_df = pd.DataFrame(luke_stats_df)
        if not luke_stats_df.empty:
            luke_stats_df = luke_stats_df.sort_values(['entity_type', 'count'], ascending=[True, False])
        
        # 保存
        luke_df.to_csv('improved_bert_ner_results/luke_only_entities_detail.csv', index=False, encoding='utf-8')
        luke_stats_df.to_csv('improved_bert_ner_results/luke_only_entity_stats.csv', index=False, encoding='utf-8')
    
    def _save_detailed_results(self):
        """詳細な分析結果を保存"""
        detailed_results = {
            'bert_extraction_details': {},
            'pattern_extraction_details': {},
            'entity_cooccurrence_matrix': {},
            'manual_section_analysis': {}
        }
        
        # BERT抽出の詳細
        for game_id, manual_data in self.results['ner_results']['manual_results'].items():
            detailed_results['bert_extraction_details'][game_id] = {
                'total_bert_entities': len(manual_data['bert_entities']),
                'bert_entities_by_type': defaultdict(int),
                'bert_entity_texts': [entity['text'] for entity in manual_data['bert_entities']]
            }
            
            for entity in manual_data['bert_entities']:
                detailed_results['bert_extraction_details'][game_id]['bert_entities_by_type'][entity['type']] += 1
        
        # パターン抽出の詳細
        for game_id, manual_data in self.results['ner_results']['manual_results'].items():
            detailed_results['pattern_extraction_details'][game_id] = {
                'pattern_entities': dict(manual_data['pattern_entities']),
                'total_pattern_entities': sum(len(entities) for entities in manual_data['pattern_entities'].values())
            }
        
        # エンティティ共起行列
        entity_list = list(self.results['ner_results']['all_entities'].keys())
        for entity_type in entity_list:
            detailed_results['entity_cooccurrence_matrix'][entity_type] = {}
            for other_type in entity_list:
                if entity_type != other_type:
                    # 簡易的な共起計算（実際の共起はネットワーク分析で詳細に計算）
                    detailed_results['entity_cooccurrence_matrix'][entity_type][other_type] = 0
        
        # マニュアルセクション分析
        for game_id, manual_data in self.results['ner_results']['manual_results'].items():
            detailed_results['manual_section_analysis'][game_id] = {
                'section_count': len(manual_data['sections']),
                'sections_with_entities': 0,
                'entity_distribution_by_section': []
            }
            
            for section in manual_data['sections']:
                section_entity_count = sum(len(entities) for entities in section['entities'].values())
                if section_entity_count > 0:
                    detailed_results['manual_section_analysis'][game_id]['sections_with_entities'] += 1
                
                detailed_results['manual_section_analysis'][game_id]['entity_distribution_by_section'].append({
                    'head': section['head'],
                    'entity_count': section_entity_count,
                    'entity_types': list(section['entities'].keys())
                })
        
        with open('improved_bert_ner_results/detailed_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2, default=str)
    
    def create_visualizations(self):
        """可視化を作成"""
        print("可視化を作成中...")
        
        # 1. エンティティ分布
        self.plot_entity_distribution()
        
        # 2. BERT vs パターン比較
        self.plot_bert_vs_pattern()
        
        # 3. ネットワーク可視化
        self.plot_network()
        
        # 4. エンティティタイプ別詳細分析
        self.plot_entity_type_analysis()
        
        # 5. マニュアル別エンティティ分布
        self.plot_manual_entity_distribution()
    
    def plot_entity_distribution(self):
        """エンティティ分布の可視化"""
        entity_stats = self.results['ner_results']['entity_statistics']
        
        plt.figure(figsize=(15, 10))
        
        # エンティティタイプ別分布
        plt.subplot(2, 2, 1)
        categories = list(entity_stats.keys())
        category_counts = [stats['unique_count'] for stats in entity_stats.values()]
        
        bars = plt.bar(categories, category_counts, color='lightblue', alpha=0.7)
        plt.title('BIOスキーマ対応LUKE抽出エンティティタイプ別分布', fontsize=14, fontweight='bold')
        plt.xlabel('エンティティタイプ')
        plt.ylabel('エンティティ数')
        plt.xticks(rotation=45)
        
        for bar, count in zip(bars, category_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom')
        
        # 上位エンティティ
        plt.subplot(2, 2, 2)
        all_entities = []
        for stats in entity_stats.values():
            all_entities.extend(stats['top_entities'][:5])
        
        if all_entities:
            entities, counts = zip(*sorted(all_entities, key=lambda x: x[1], reverse=True)[:10])
            bars = plt.barh(range(len(entities)), counts, color='lightcoral', alpha=0.7)
            plt.yticks(range(len(entities)), entities)
            plt.title('上位エンティティ（BIOスキーマ対応LUKE抽出）', fontsize=14, fontweight='bold')
            plt.xlabel('出現回数')
        
        # ネットワーク統計
        plt.subplot(2, 2, 3)
        network_stats = self.results['network_analysis']['basic_stats']
        if network_stats:
            stats_names = ['ノード数', 'エッジ数', '密度', 'クラスタリング係数']
            stats_values = [
                network_stats['nodes'],
                network_stats['edges'],
                network_stats['density'],
                network_stats['average_clustering']
            ]
            
            bars = plt.bar(stats_names, stats_values, color='lightgreen', alpha=0.7)
            plt.title('ネットワーク統計（BIOスキーマ対応LUKE抽出）', fontsize=14, fontweight='bold')
            plt.ylabel('値')
            plt.xticks(rotation=45)
        
        # 中心性分布
        plt.subplot(2, 2, 4)
        centrality = self.results['network_analysis']['centrality']
        if centrality and 'degree' in centrality:
            degree_values = list(centrality['degree'].values())
            plt.hist(degree_values, bins=20, color='lightyellow', alpha=0.7, edgecolor='black')
            plt.title('次数中心性分布', fontsize=14, fontweight='bold')
            plt.xlabel('次数中心性')
            plt.ylabel('頻度')
        
        plt.tight_layout()
        plt.savefig('improved_bert_ner_results/improved_luke_entity_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_bert_vs_pattern(self):
        """BERT vs パターン抽出の比較"""
        manual_data = self.results['ner_results']['manual_results']
        
        # ラベルが長いため、縦配置で見切れないように図を拡大
        plt.figure(figsize=(16, 10))
        
        bert_counts = []
        pattern_counts = []
        manual_names = []
        
        for game_id, data in manual_data.items():
            manual_names.append(data['title'])
            bert_counts.append(len(data['bert_entities']))
            pattern_counts.append(sum(len(entities) for entities in data['pattern_entities'].values()))
        
        x = np.arange(len(manual_names))
        width = 0.35
        
        plt.bar(x - width/2, bert_counts, width, label='LUKE抽出', color='lightblue', alpha=0.7)
        plt.bar(x + width/2, pattern_counts, width, label='パターン抽出', color='lightcoral', alpha=0.7)
        
        plt.title('BIOスキーマ対応LUKE vs パターン抽出の比較', fontsize=16, fontweight='bold')
        plt.xlabel('マニュアル')
        plt.ylabel('エンティティ数')
        # X軸ラベルを90度回転（縦）し、余白を確保
        plt.xticks(x, manual_names, rotation=90, ha='center', va='top')
        plt.subplots_adjust(bottom=0.35)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig('improved_bert_ner_results/improved_luke_vs_pattern_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_network(self):
        """ネットワークの可視化"""
        graph = self.results['networks']['entity_network']
        
        if len(graph.nodes()) == 0:
            print("ネットワークにノードが存在しません")
            return
        
        plt.figure(figsize=(16, 12))
        
        # ネットワークレイアウト
        pos = nx.spring_layout(graph, k=1, iterations=50)
        
        # ノードのサイズを次数に基づいて設定
        node_sizes = [graph.degree(node) * 100 for node in graph.nodes()]
        
        # ネットワークを描画
        nx.draw_networkx_nodes(graph, pos,
                             node_size=node_sizes,
                             node_color='lightblue',
                             alpha=0.7)
        
        nx.draw_networkx_edges(graph, pos,
                             alpha=0.5,
                             edge_color='gray')
        
        # ラベルを描画（ノード数が少ない場合のみ）
        if len(graph.nodes()) <= 50:
            nx.draw_networkx_labels(graph, pos,
                                  font_size=8)
        
        plt.title('BIOスキーマ対応LUKE抽出エンティティネットワーク', fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        plt.savefig('improved_bert_ner_results/improved_luke_network.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_entity_type_analysis(self):
        """エンティティタイプ別詳細分析の可視化"""
        entity_stats = self.results['ner_results']['entity_statistics']
        
        plt.figure(figsize=(16, 12))
        
        # エンティティタイプ別の詳細統計
        plt.subplot(2, 3, 1)
        categories = list(entity_stats.keys())
        total_counts = [stats['total_count'] for stats in entity_stats.values()]
        unique_counts = [stats['unique_count'] for stats in entity_stats.values()]
        
        x = np.arange(len(categories))
        width = 0.35
        
        plt.bar(x - width/2, total_counts, width, label='総出現回数', color='lightblue', alpha=0.7)
        plt.bar(x + width/2, unique_counts, width, label='ユニーク数', color='lightcoral', alpha=0.7)
        plt.title('エンティティタイプ別詳細統計', fontsize=14, fontweight='bold')
        plt.xlabel('エンティティタイプ')
        plt.ylabel('数')
        plt.xticks(x, categories, rotation=45)
        plt.legend()
        
        # エンティティタイプ別の頻度分布
        plt.subplot(2, 3, 2)
        for i, (entity_type, stats) in enumerate(entity_stats.items()):
            if stats['top_entities']:
                entities, counts = zip(*stats['top_entities'][:5])
                plt.barh(range(len(entities)), counts, 
                        label=entity_type, alpha=0.7)
                plt.yticks(range(len(entities)), entities)
        
        plt.title('エンティティタイプ別上位エンティティ', fontsize=14, fontweight='bold')
        plt.xlabel('出現回数')
        plt.legend()
        
        # エンティティ長分布
        plt.subplot(2, 3, 3)
        entity_lengths = []
        entity_types = []
        
        for entity_type, entities in self.results['ner_results']['all_entities'].items():
            for entity in entities:
                entity_lengths.append(len(entity))
                entity_types.append(entity_type)
        
        if entity_lengths:
            plt.hist(entity_lengths, bins=20, alpha=0.7, edgecolor='black')
            plt.title('エンティティ長分布', fontsize=14, fontweight='bold')
            plt.xlabel('エンティティ長（文字数）')
            plt.ylabel('頻度')
        
        # エンティティタイプ別の平均長
        plt.subplot(2, 3, 4)
        avg_lengths = {}
        for entity_type, entities in self.results['ner_results']['all_entities'].items():
            if entities:
                avg_lengths[entity_type] = np.mean([len(entity) for entity in entities])
        
        if avg_lengths:
            plt.bar(avg_lengths.keys(), avg_lengths.values(), color='lightgreen', alpha=0.7)
            plt.title('エンティティタイプ別平均長', fontsize=14, fontweight='bold')
            plt.xlabel('エンティティタイプ')
            plt.ylabel('平均長（文字数）')
            plt.xticks(rotation=45)
        
        # ネットワーク中心性分布
        plt.subplot(2, 3, 5)
        centrality = self.results['network_analysis']['centrality']
        if centrality and 'betweenness' in centrality:
            betweenness_values = list(centrality['betweenness'].values())
            plt.hist(betweenness_values, bins=20, color='lightyellow', alpha=0.7, edgecolor='black')
            plt.title('媒介中心性分布', fontsize=14, fontweight='bold')
            plt.xlabel('媒介中心性')
            plt.ylabel('頻度')
        
        # クラスタリング係数分布
        plt.subplot(2, 3, 6)
        graph = self.results['networks']['entity_network']
        if len(graph.nodes()) > 0:
            clustering_coeffs = list(nx.clustering(graph).values())
            plt.hist(clustering_coeffs, bins=20, color='lightpink', alpha=0.7, edgecolor='black')
            plt.title('クラスタリング係数分布', fontsize=14, fontweight='bold')
            plt.xlabel('クラスタリング係数')
            plt.ylabel('頻度')
        
        plt.tight_layout()
        plt.savefig('improved_bert_ner_results/improved_luke_entity_type_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_manual_entity_distribution(self):
        """マニュアル別エンティティ分布の可視化"""
        manual_data = self.results['ner_results']['manual_results']
        
        plt.figure(figsize=(16, 10))
        
        # マニュアル別エンティティ数
        plt.subplot(2, 2, 1)
        manual_names = []
        total_entities = []
        bert_entities = []
        pattern_entities = []
        
        for game_id, data in manual_data.items():
            manual_names.append(data['title'])
            total_entity_count = sum(len(entities) for entities in data['entities'].values())
            total_entities.append(total_entity_count)
            bert_entities.append(len(data['bert_entities']))
            pattern_entities.append(sum(len(entities) for entities in data['pattern_entities'].values()))
        
        x = np.arange(len(manual_names))
        width = 0.25
        
        plt.bar(x - width, total_entities, width, label='総エンティティ', color='lightblue', alpha=0.7)
        plt.bar(x, bert_entities, width, label='LUKE抽出', color='lightcoral', alpha=0.7)
        plt.bar(x + width, pattern_entities, width, label='パターン抽出', color='lightgreen', alpha=0.7)
        
        plt.title('マニュアル別エンティティ数比較', fontsize=14, fontweight='bold')
        plt.xlabel('マニュアル')
        plt.ylabel('エンティティ数')
        plt.xticks(x, manual_names, rotation=45)
        plt.legend()
        
        # マニュアル別エンティティタイプ分布
        plt.subplot(2, 2, 2)
        entity_types = list(self.results['ner_results']['all_entities'].keys())
        manual_entity_matrix = []
        
        for game_id, data in manual_data.items():
            row = []
            for entity_type in entity_types:
                row.append(len(data['entities'].get(entity_type, [])))
            manual_entity_matrix.append(row)
        
        if manual_entity_matrix:
            im = plt.imshow(manual_entity_matrix, cmap='YlOrRd', aspect='auto')
            plt.colorbar(im)
            plt.title('マニュアル別エンティティタイプ分布', fontsize=14, fontweight='bold')
            plt.xlabel('エンティティタイプ')
            plt.ylabel('マニュアル')
            plt.xticks(range(len(entity_types)), entity_types, rotation=45)
            plt.yticks(range(len(manual_names)), manual_names)
        
        # セクション別エンティティ分布
        plt.subplot(2, 2, 3)
        section_entity_counts = []
        section_names = []
        
        for game_id, data in manual_data.items():
            for section in data['sections']:
                section_entity_count = sum(len(entities) for entities in section['entities'].values())
                if section_entity_count > 0:
                    section_entity_counts.append(section_entity_count)
                    section_names.append(f"{data['title']}: {section['head']}")
        
        if section_entity_counts:
            plt.hist(section_entity_counts, bins=20, color='lightyellow', alpha=0.7, edgecolor='black')
            plt.title('セクション別エンティティ数分布', fontsize=14, fontweight='bold')
            plt.xlabel('エンティティ数')
            plt.ylabel('セクション数')
        
        # エンティティ密度（マニュアルあたりのエンティティ数）
        plt.subplot(2, 2, 4)
        entity_densities = []
        for game_id, data in manual_data.items():
            total_entities = sum(len(entities) for entities in data['entities'].values())
            total_sections = len(data['sections'])
            if total_sections > 0:
                density = total_entities / total_sections
                entity_densities.append(density)
        
        if entity_densities:
            plt.hist(entity_densities, bins=10, color='lightpink', alpha=0.7, edgecolor='black')
            plt.title('マニュアル別エンティティ密度分布', fontsize=14, fontweight='bold')
            plt.xlabel('エンティティ密度（セクションあたり）')
            plt.ylabel('マニュアル数')
        
        plt.tight_layout()
        plt.savefig('improved_bert_ner_results/improved_luke_manual_entity_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """メイン関数"""
    if not TRANSFORMERS_AVAILABLE:
        print("エラー: transformersライブラリが必要です。")
        print("インストール方法: pip install transformers torch datasets")
        return
    
    print("BIOスキーマ対応LUKEベースの固有表現抽出を開始します...")
    
    # デフォルトモデルで実行
    model_name = "studio-ousia/luke-japanese-base-lite"
    print(f"使用モデル: {model_name}")
    
    # 設定オプション
    train_model = False  # ファインチューニングを無効化（まずベースモデルをテスト）
    use_annotated_data = True  # アノテーションデータを使用
    
    try:
        analyzer = ImprovedBERTNERAnalyzer(model_name)
        results = analyzer.analyze(train_model=train_model, use_annotated_data=use_annotated_data)
        
        # 結果の要約を表示
        print("\n=== BIOスキーマ対応LUKE分析結果要約 ===")
        
        # エンティティ統計
        entity_stats = results['ner_results']['entity_statistics']
        total_entities = sum(stats['total_count'] for stats in entity_stats.values())
        print(f"総エンティティ数: {total_entities}")
        
        for entity_type, stats in entity_stats.items():
            print(f"{entity_type}: {stats['unique_count']}種類 ({stats['total_count']}回出現)")
        
        # ネットワーク統計
        network_stats = results['network_analysis']['basic_stats']
        if network_stats:
            print(f"\nネットワーク統計:")
            print(f"ノード数: {network_stats['nodes']}")
            print(f"エッジ数: {network_stats['edges']}")
            print(f"密度: {network_stats['density']:.3f}")
            print(f"平均クラスタリング係数: {network_stats['average_clustering']:.3f}")
        
        # 上位エンティティ
        centrality = results['network_analysis']['centrality']
        if centrality and 'degree' in centrality:
            print(f"\n上位エンティティ（次数中心性）:")
            top_entities = sorted(centrality['degree'].items(), key=lambda x: x[1], reverse=True)[:10]
            for i, (entity, centrality_value) in enumerate(top_entities, 1):
                print(f"{i}. {entity}: {centrality_value:.3f}")
        
        # 分析完了メッセージ
        print(f"\n分析が完了しました！")
        print(f"結果ファイル:")
        print(f"- improved_bert_ner_results/improved_luke_ner_results.json")
        print(f"- improved_bert_ner_results/improved_luke_entity_list.csv")
        print(f"- improved_bert_ner_results/detailed_analysis.json")
        print(f"可視化ファイル:")
        print(f"- improved_bert_ner_results/improved_luke_entity_distribution.png")
        print(f"- improved_bert_ner_results/improved_luke_vs_pattern_comparison.png")
        print(f"- improved_bert_ner_results/improved_luke_network.png")
        print(f"- improved_bert_ner_results/improved_luke_entity_type_analysis.png")
        print(f"- improved_bert_ner_results/improved_luke_manual_entity_distribution.png")
        
        # LUKE専用統計の表示
        luke_entity_count = sum(len(manual_data['bert_entities']) for manual_data in results['ner_results']['manual_results'].values())
        pattern_entity_count = sum(sum(len(entities) for entities in manual_data['pattern_entities'].values()) for manual_data in results['ner_results']['manual_results'].values())
        
        print(f"\n=== 抽出手法別統計 ===")
        print(f"LUKE抽出: {luke_entity_count}件")
        print(f"パターン抽出: {pattern_entity_count}件")
        print(f"総エンティティ数: {total_entities}")
        print(f"LUKE割合: {luke_entity_count/total_entities*100:.1f}%")
        print(f"パターン割合: {pattern_entity_count/total_entities*100:.1f}%")
        
    except Exception as e:
        print(f"分析中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 