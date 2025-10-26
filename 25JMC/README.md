# じんもんこん2025 研究用オープンデータ・コード

人文科学とコンピュータシンポジウム（じんもんこん2025）の論文「ゲームマニュアルにおける没入的記述の構造化分析—大規模言語モデルを用いた自動セクション分類—」で用いるオープンデータとコード一式です。

## 📂 ファイル構成

### 1. データ定義・指示書

#### `game_manual_schema.xsd`
ゲームマニュアルXMLの構造を定義するXMLスキーマファイル

#### `prompt.txt`
claude-4-sonnetにXMLマークアップを依頼する際のプロンプト


---

### 2. OCR・テキスト抽出

#### `gcloudocr.py`
Google Cloud Vision APIを使用したシンプルなOCRスクリプト。日本語のルビ（ふりがな）を自動除去します。

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

**出力ファイル:**
- `comparison_report_{game_id}.txt`: 各ゲームの詳細レポート
- `comparison_summary_report.txt`: 全ゲームの統合レポート

---

### 4. 説明・没入セクション分析

#### `analyze_immersion_instruction.py`
ゲームマニュアルのセクションを「説明的記述」と「没入的記述」に分類し、文字数や割合を分析します。

---

### 5. ナラティブ・クラスタリング分析

#### `game_narrative_clustering.py`
ゲームマニュアルのナラティブ（物語）記述をBERTエンコーディングでベクトル化し、クラスタリング分析を行います。

**モデル:**
- `cl-tohoku/bert-base-japanese-whole-word-masking`
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

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

---

**最終更新**: 2025年10月  
**バージョン**: 1.0  

