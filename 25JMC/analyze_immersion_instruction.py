#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ゲームマニュアルの説明・没入セクション分析スクリプト
"""

import os
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
import re
import json
import csv
from datetime import datetime

# 説明・没入の対応関係
IMMERSION_VS_INSTRUCTION = {
    'gameplay': '没入',
    'controls': '説明',
    'control': '説明',
    'character': '没入',
    'item': '没入',
    'system': '説明',
    'narrative': '没入',
    'instruction': '説明',
    'reference': '説明',
    'warning': '説明',
    'legal': '説明',
    'enemy': '没入',
    'cover': '説明',
    'preface': '説明',
    'greeting': '説明',
    'toc': '説明',
    'company_info': '説明',
    'blank': '説明',
    'empty': '説明',
    'memo': '説明'
}

def count_characters(text):
    """テキストの文字数をカウント（空白・改行を除く）"""
    if not text:
        return 0
    # 空白、改行、タブを除去して文字数をカウント
    return len(re.sub(r'\s+', '', text))

def analyze_xml_file(file_path):
    """単一のXMLファイルを分析"""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # 統計データ
        stats = {
            '説明': {'count': 0, 'char_count': 0, 'sections': []},
            '没入': {'count': 0, 'char_count': 0, 'sections': []},
            '未分類': {'count': 0, 'char_count': 0, 'sections': []}
        }
        
        # すべてのsection要素を取得
        sections = root.findall('.//section')
        
        for section in sections:
            section_type = section.get('type', '')
            
            # セクション内のテキストを収集
            text_content = []
            
            # head要素のテキスト
            head_elem = section.find('head')
            if head_elem is not None and head_elem.text:
                text_content.append(head_elem.text)
            
            # p要素のテキスト
            for p_elem in section.findall('p'):
                if p_elem.text:
                    text_content.append(p_elem.text)
            
            # テキストを結合
            full_text = ' '.join(text_content)
            char_count = count_characters(full_text)
            
            # 分類
            if section_type in IMMERSION_VS_INSTRUCTION:
                category = IMMERSION_VS_INSTRUCTION[section_type]
                stats[category]['count'] += 1
                stats[category]['char_count'] += char_count
                stats[category]['sections'].append({
                    'type': section_type,
                    'char_count': char_count,
                    'text_preview': full_text[:100] + '...' if len(full_text) > 100 else full_text
                })
            else:
                stats['未分類']['count'] += 1
                stats['未分類']['char_count'] += char_count
                stats['未分類']['sections'].append({
                    'type': section_type,
                    'char_count': char_count,
                    'text_preview': full_text[:100] + '...' if len(full_text) > 100 else full_text
                })
        
        return stats
        
    except Exception as e:
        print(f"エラー: {file_path} の処理中にエラーが発生しました: {e}")
        return None

def analyze_all_files(folder_path):
    """フォルダ内のすべてのXMLファイルを分析"""
    all_stats = {
        '説明': {'count': 0, 'char_count': 0, 'sections': []},
        '没入': {'count': 0, 'char_count': 0, 'sections': []},
        '未分類': {'count': 0, 'char_count': 0, 'sections': []}
    }
    
    file_stats = {}
    processed_files = 0
    
    # XMLファイルを検索
    for filename in os.listdir(folder_path):
        if filename.endswith('.xml'):
            file_path = os.path.join(folder_path, filename)
            print(f"処理中: {filename}")
            
            stats = analyze_xml_file(file_path)
            if stats:
                file_stats[filename] = stats
                processed_files += 1
                
                # 全体統計に追加
                for category in ['説明', '没入', '未分類']:
                    all_stats[category]['count'] += stats[category]['count']
                    all_stats[category]['char_count'] += stats[category]['char_count']
                    all_stats[category]['sections'].extend(stats[category]['sections'])
    
    return all_stats, file_stats, processed_files

def print_results(all_stats, file_stats, processed_files):
    """結果を表示"""
    print("\n" + "="*60)
    print("ゲームマニュアル 説明・没入セクション分析結果")
    print("="*60)
    
    print(f"\n処理ファイル数: {processed_files}")
    
    print("\n【全体統計】")
    print("-" * 40)
    
    for category in ['説明', '没入', '未分類']:
        stats = all_stats[category]
        print(f"{category}:")
        print(f"  セクション数: {stats['count']:,}")
        print(f"  文字数: {stats['char_count']:,}")
        if stats['count'] > 0:
            avg_chars = stats['char_count'] / stats['count']
            print(f"  平均文字数: {avg_chars:.1f}")
        print()
    
    # 合計
    total_sections = sum(all_stats[cat]['count'] for cat in ['説明', '没入', '未分類'])
    total_chars = sum(all_stats[cat]['char_count'] for cat in ['説明', '没入', '未分類'])
    
    print(f"合計セクション数: {total_sections:,}")
    print(f"合計文字数: {total_chars:,}")
    
    # 割合
    if total_sections > 0:
        print(f"\n【割合】")
        print("-" * 40)
        for category in ['説明', '没入', '未分類']:
            section_ratio = all_stats[category]['count'] / total_sections * 100
            char_ratio = all_stats[category]['char_count'] / total_chars * 100
            print(f"{category}: セクション {section_ratio:.1f}%, 文字数 {char_ratio:.1f}%")
    
    # ファイル別統計
    print(f"\n【ファイル別統計】")
    print("-" * 40)
    for filename, stats in sorted(file_stats.items()):
        total_file_sections = sum(stats[cat]['count'] for cat in ['説明', '没入', '未分類'])
        if total_file_sections > 0:
            immersion_ratio = stats['没入']['count'] / total_file_sections * 100
            instruction_ratio = stats['説明']['count'] / total_file_sections * 100
            print(f"{filename}: 没入 {immersion_ratio:.1f}%, 説明 {instruction_ratio:.1f}%")

def save_results_to_json(all_stats, file_stats, processed_files, output_dir="output"):
    """分析結果をJSONファイルに保存"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"immersion_instruction_analysis_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # 結果データを構造化
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'processed_files': processed_files,
            'analysis_type': 'immersion_instruction_analysis'
        },
        'overall_stats': all_stats,
        'file_stats': file_stats,
        'summary': {}
    }
    
    # サマリー情報を追加
    total_sections = sum(all_stats[cat]['count'] for cat in ['説明', '没入', '未分類'])
    total_chars = sum(all_stats[cat]['char_count'] for cat in ['説明', '没入', '未分類'])
    
    results['summary'] = {
        'total_sections': total_sections,
        'total_characters': total_chars,
        'ratios': {}
    }
    
    if total_sections > 0:
        for category in ['説明', '没入', '未分類']:
            section_ratio = all_stats[category]['count'] / total_sections * 100
            char_ratio = all_stats[category]['char_count'] / total_chars * 100
            results['summary']['ratios'][category] = {
                'section_ratio': section_ratio,
                'char_ratio': char_ratio
            }
    
    # JSONファイルに保存
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nJSON結果を保存しました: {filepath}")
    return filepath

def save_results_to_csv(all_stats, file_stats, processed_files, output_dir="output"):
    """分析結果をCSVファイルに保存"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 全体統計CSV
    overall_filename = f"overall_stats_{timestamp}.csv"
    overall_filepath = os.path.join(output_dir, overall_filename)
    
    with open(overall_filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['カテゴリ', 'セクション数', '文字数', '平均文字数'])
        
        for category in ['説明', '没入', '未分類']:
            stats = all_stats[category]
            avg_chars = stats['char_count'] / stats['count'] if stats['count'] > 0 else 0
            writer.writerow([category, stats['count'], stats['char_count'], f"{avg_chars:.1f}"])
    
    # 2. ファイル別統計CSV
    file_stats_filename = f"file_stats_{timestamp}.csv"
    file_stats_filepath = os.path.join(output_dir, file_stats_filename)
    
    with open(file_stats_filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ファイル名', '説明_セクション数', '説明_文字数', '没入_セクション数', '没入_文字数', '未分類_セクション数', '未分類_文字数', '没入_割合(%)', '説明_割合(%)'])
        
        for filename, stats in sorted(file_stats.items()):
            total_file_sections = sum(stats[cat]['count'] for cat in ['説明', '没入', '未分類'])
            immersion_ratio = stats['没入']['count'] / total_file_sections * 100 if total_file_sections > 0 else 0
            instruction_ratio = stats['説明']['count'] / total_file_sections * 100 if total_file_sections > 0 else 0
            
            writer.writerow([
                filename,
                stats['説明']['count'],
                stats['説明']['char_count'],
                stats['没入']['count'],
                stats['没入']['char_count'],
                stats['未分類']['count'],
                stats['未分類']['char_count'],
                f"{immersion_ratio:.1f}",
                f"{instruction_ratio:.1f}"
            ])
    
    # 3. セクション詳細CSV
    section_details_filename = f"section_details_{timestamp}.csv"
    section_details_filepath = os.path.join(output_dir, section_details_filename)
    
    with open(section_details_filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['カテゴリ', 'セクションタイプ', '文字数', 'テキストプレビュー'])
        
        for category in ['説明', '没入', '未分類']:
            for section in all_stats[category]['sections']:
                writer.writerow([
                    category,
                    section['type'],
                    section['char_count'],
                    section['text_preview']
                ])
    
    print(f"\nCSV結果を保存しました:")
    print(f"  全体統計: {overall_filepath}")
    print(f"  ファイル別統計: {file_stats_filepath}")
    print(f"  セクション詳細: {section_details_filepath}")
    
    return overall_filepath, file_stats_filepath, section_details_filepath

def main():
    """メイン関数"""
    folder_path = "../game_manual/cursor_labeled_integrated"
    
    if not os.path.exists(folder_path):
        print(f"エラー: フォルダ '{folder_path}' が見つかりません。")
        return
    
    print("XMLファイルの分析を開始します...")
    all_stats, file_stats, processed_files = analyze_all_files(folder_path)
    
    if processed_files == 0:
        print("処理されたファイルがありません。")
        return
    
    print_results(all_stats, file_stats, processed_files)
    
    # 詳細なセクションタイプ分析
    print(f"\n【セクションタイプ詳細】")
    print("-" * 40)
    
    type_counter = Counter()
    for category in ['説明', '没入', '未分類']:
        for section in all_stats[category]['sections']:
            type_counter[section['type']] += 1
    
    for section_type, count in type_counter.most_common():
        category = IMMERSION_VS_INSTRUCTION.get(section_type, '未分類')
        print(f"{section_type}: {count} ({category})")
    
    # 結果をファイルに保存
    print(f"\n" + "="*60)
    print("結果をファイルに保存中...")
    print("="*60)
    
    # JSONファイルに保存
    json_filepath = save_results_to_json(all_stats, file_stats, processed_files)
    
    # CSVファイルに保存
    csv_filepaths = save_results_to_csv(all_stats, file_stats, processed_files)
    
    print(f"\n分析完了！結果は以下のファイルに保存されました:")
    print(f"JSON: {json_filepath}")
    print(f"CSV: {csv_filepaths[0]}, {csv_filepaths[1]}, {csv_filepaths[2]}")

if __name__ == "__main__":
    main() 