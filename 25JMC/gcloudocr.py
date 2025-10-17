import os
import io
from google.cloud import vision

# Google Vision API クライアントを作成
client = vision.ImageAnnotatorClient()

# 入力・出力フォルダの設定
image_dir = "output_images"
output_dir = "output_texts"
os.makedirs(output_dir, exist_ok=True)

def format_text(lines):
    """改行を適切に整理し、段落ごとにまとめる"""
    paragraphs = []
    current_paragraph = []

    for line in lines:
        if line.strip():  # 空行でなければ
            current_paragraph.append(line.strip())
        else:  # 空行なら段落を確定
            if current_paragraph:
                paragraphs.append("".join(current_paragraph))  # 改行を削除して結合
                current_paragraph = []

    if current_paragraph:  # 最後の段落を追加
        paragraphs.append("".join(current_paragraph))

    return "\n\n".join(paragraphs)  # 段落ごとに改行を挿入

def extract_text_from_image(image_path):
    """Google Cloud Vision OCRで画像からテキストを抽出し、ルビを除去"""
    with io.open(image_path, "rb") as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)

    if response.error.message:
        print(f"Error processing {image_path}: {response.error.message}")
        return ""

    paragraphs = []

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            block_text = []

            for paragraph in block.paragraphs:
                paragraph_text = []

                for word in paragraph.words:
                    word_text = "".join([symbol.text for symbol in word.symbols])
                    font_size = word.bounding_box.vertices[2].y - word.bounding_box.vertices[0].y  # 縦の長さ

                    if font_size > 10:  # ルビ除去
                        paragraph_text.append(word_text)

                if paragraph_text:
                    block_text.append(" ".join(paragraph_text))  # 単語をスペースで結合してパラグラフに

            if block_text:
                paragraphs.append("\n".join(block_text))  # パラグラフ間は改行で区切る

    return "\n\n".join(paragraphs)  # ブロック間は空行で区切る

# 画像フォルダ内の全画像を処理
for filename in sorted(os.listdir(image_dir)):  # ソートして順番に処理
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")):
        image_path = os.path.join(image_dir, filename)
        
        # OCR実行
        extracted_text = extract_text_from_image(image_path)
        
        # テキストを保存
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(extracted_text)

        print(f"Processed: {filename} → {output_path}")
