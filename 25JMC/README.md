# じんもんこん2025 研究用オープンデータ・コード

人文科学とコンピュータシンポジウム（じんもんこん2025）の論文で用いるオープンデータとコード一式です。

## 📂 ファイル構成

### 1. データ定義・指示書

#### `game_manual_schema.xsd`
ゲームマニュアルXMLの構造を定義するXMLスキーマファイルです。

#### `prompt.txt`
Gemini/ChatGPT等のLLMにXMLマークアップを依頼する際のプロンプトです。

**使用方法:**
1. このプロンプトをLLMに送信
2. OCRで抽出したテキストとXSDファイルを添付
3. LLMが自動的にXML形式でマークアップ

**指示内容:**
- XSDスキーマに従ったマークアップ
- 元のテキストは編集しない（OCRミスの改行のみ修正可）
- セクション・サブセクションにtype属性を付与
- 見出しがない場合はhead要素を省略

---

### 2. OCR・テキスト抽出

#### `gcloudocr.py`
Google Cloud Vision APIを使用したシンプルなOCRスクリプト。日本語のルビ（ふりがな）を自動除去します。

**主な機能:**
- Vision APIによる高精度OCR
- フォントサイズベースのルビ除去（10ピクセル未満を除外）
- 段落・ブロック構造の保持
- バッチ処理対応

**使用方法:**
```python
python gcloudocr.py
```

**設定:**
- 入力: `output_images/` ディレクトリの画像
- 出力: `output_texts/` ディレクトリのテキストファイル

**必要な環境変数:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/keyfile.json"
```

---

### 3. XML構造比較・評価

#### `xml_batch_comparison.py`
AI生成XMLと人手作成XMLの構造を一括比較し、精度を評価するツールです。

**評価指標:**
- **F1スコア**: セクション分類の精度
- **ツリー編集距離**: 構造の類似度
- **Jaccard類似度**: セクションタイプの一致度
- **Precision/Recall**: 分類の適合率・再現率

**使用方法:**
```bash
# デフォルト設定で実行（自動検出）
python xml_batch_comparison.py

# カスタムディレクトリ指定
python xml_batch_comparison.py \
    --ai-dir cursor_labeled_integrated \
    --human-dir output_images_xml_golden

# 特定のゲームIDのみ比較
python xml_batch_comparison.py \
    --game-ids CLV-P-HAAAJ CLV-P-HABCJ
```

**出力ファイル:**
- `comparison_report_{game_id}.txt`: 各ゲームの詳細レポート
- `comparison_summary_report.txt`: 全ゲームの統合レポート

**レポート内容:**
```
📊 PER-GAME SUMMARY:
Game ID              AI Secs    Hum Secs   F1-Score   Edit Dist   
--------------------------------------------------------------------------------
CLV-P-HAAAJ          45         48         0.875      0.125       
CLV-P-HABCJ          52         50         0.920      0.080       

AVERAGE                                    0.898      0.103       

💡 OVERALL ASSESSMENT:
  ✅ 優秀: AI生成のセクション構造は全体的に非常に高品質
```

---

### 4. 説明・没入セクション分析

#### `analyze_immersion_instruction.py`
ゲームマニュアルのセクションを「説明的記述」と「没入的記述」に分類し、文字数や割合を分析します。

**分類基準:**

| カテゴリ | セクションタイプ | 例 |
|---------|-----------------|-----|
| **没入的** | gameplay, narrative, character, item, enemy | ゲームプレイ、ストーリー、キャラクター説明 |
| **説明的** | instruction, control, system, warning, legal | 操作方法、システム説明、注意事項 |

**使用方法:**
```bash
python analyze_immersion_instruction.py
```

**出力ファイル（output/ディレクトリ）:**
1. `immersion_instruction_analysis_{timestamp}.json`: 詳細な分析結果
2. `overall_stats_{timestamp}.csv`: 全体統計
3. `file_stats_{timestamp}.csv`: ファイル別統計
4. `section_details_{timestamp}.csv`: セクション詳細

**分析結果例:**
```
【全体統計】
説明:
  セクション数: 497
  文字数: 125,430
  平均文字数: 252.4

没入:
  セクション数: 384
  文字数: 98,760
  平均文字数: 257.2

【割合】
説明: セクション 56.4%, 文字数 55.9%
没入: セクション 43.6%, 文字数 44.1%
```

---

### 5. ナラティブ・クラスタリング分析

#### `game_narrative_clustering.py`
ゲームマニュアルのナラティブ（物語）記述をBERTエンコーディングでベクトル化し、クラスタリング分析を行います。

**主な機能:**
- **BERTエンコーディング**: 日本語BERTモデルでテキストをベクトル化
- **次元削減**: PCA、t-SNE、UMAPによる可視化
- **クラスタリング**: K-Means、DBSCANによるグループ化
- **キーワード抽出**: TF-IDFによるクラスタ特徴抽出

**使用するモデル:**
- `cl-tohoku/bert-base-japanese-whole-word-masking`
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

**使用方法:**
```python
python game_narrative_clustering.py
```

**出力:**
- クラスタリング結果の可視化（散布図）
- 各クラスタの代表的なテキスト
- クラスタごとのキーワード

**分析項目:**
- ゲームジャンルによるナラティブの特徴
- ストーリー記述の類似性
- 時代による記述スタイルの変化

---

### 6. BERT固有表現認識（NER）

#### `improved_bert_ner.py`
BERTモデルを使用した固有表現認識（Named Entity Recognition）システムです。ゲームマニュアルから重要な情報を自動抽出します。

**抽出対象エンティティ:**
- **CHARACTER**: キャラクター名
- **ITEM**: アイテム、武器、道具
- **LOCATION**: 場所、ステージ、ワールド
- **ENEMY**: 敵キャラクター、ボス
- **ACTION**: 操作、アクション
- **GAME_ELEMENT**: ゲーム要素、システム

**使用モデル:**
- `studio-ousia/luke-japanese-base-lite`
- `cl-tohoku/bert-base-japanese-whole-word-masking`

**使用方法:**
```python
from improved_bert_ner import ImprovedBERTNERExtractor

# NER抽出器を初期化
extractor = ImprovedBERTNERExtractor()

# テキストからエンティティを抽出
text = "マリオがクッパ城でピーチ姫を助ける"
entities = extractor.extract_entities(text)

# 結果: [('マリオ', 'CHARACTER'), ('クッパ城', 'LOCATION'), ('ピーチ姫', 'CHARACTER')]
```

**機能:**
- BIOスキーマによるトークン分類
- ファインチューニング対応
- アノテーション済みデータでの学習
- 共起ネットワーク分析
- エンティティ統計の可視化

## 🎯 研究の流れ

### 1. データ収集・前処理
```bash
# ステップ1: PDF→画像変換（別ツール）
# ステップ2: OCRでテキスト抽出
python gcloudocr.py
```

### 2. LLM生成データと人手のゴールドデータのXMLセクション分類の比較
```bash
# ステップ3: LLMでXMLマークアップ（prompt.txtを使用）
# ステップ4: AI生成XMLと人手XMLを比較
python xml_batch_comparison.py
```

### 3. テキスト分析
```bash
# ステップ5: 説明・没入セクション分析
python analyze_immersion_instruction.py

# ステップ6: ナラティブクラスタリング
python game_narrative_clustering.py

# ステップ7: 固有表現抽出
python improved_bert_ner.py
```


---


## 📞 お問い合わせ

本レポジトリのコードは、MITライセンスで公開します。

---

## ⚖️ ライセンス

本研究は学術研究目的で公開されています。

---

**最終更新**: 2025年10月  
**バージョン**: 1.0  

