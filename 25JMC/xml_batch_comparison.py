"""
XML Batch Comparison Tool
複数のゲームマニュアルXMLを一括比較
"""

import sys
import os
from pathlib import Path
from xml_section_comparison import XMLSectionComparison
from typing import List, Dict


class XMLBatchComparison:
    """複数XMLファイルの一括比較クラス"""
    
    def __init__(self, ai_dir: str, human_dir: str, game_ids: List[str] = None):
        self.ai_dir = Path(ai_dir)
        self.human_dir = Path(human_dir)
        self.game_ids = game_ids if game_ids else self._auto_detect_game_ids()
        self.results = []
    
    def _auto_detect_game_ids(self) -> List[str]:
        """human_dirから_human.xmlファイルを検出してゲームIDリストを生成"""
        print(f"🔍 Auto-detecting game IDs from: {self.human_dir}")
        
        game_ids = []
        for file_path in self.human_dir.glob("*_human.xml"):
            # ファイル名から_human.xmlを除いてゲームIDを取得
            game_id = file_path.stem.replace('_human', '')
            
            # 対応するAIファイルが存在するか確認
            ai_file = self.ai_dir / f"{game_id}_labeled.xml"
            if ai_file.exists():
                game_ids.append(game_id)
                print(f"  ✅ Found: {game_id}")
            else:
                print(f"  ⚠️  Skipped: {game_id} (no AI file)")
        
        print(f"\n📊 Total games found: {len(game_ids)}")
        return sorted(game_ids)
        
    def run_batch_comparison(self):
        """バッチ比較を実行"""
        print("=" * 80)
        print("XML BATCH COMPARISON - Multiple Game Manuals")
        print("=" * 80)
        print(f"\nComparing {len(self.game_ids)} game manual(s)...\n")
        
        for game_id in self.game_ids:
            print(f"\n{'='*80}")
            print(f"Processing: {game_id}")
            print(f"{'='*80}")
            
            # ファイルパスを構築
            ai_file = self.ai_dir / f"{game_id}_labeled.xml"
            human_file = self.human_dir / f"{game_id}_human.xml"
            
            # ファイルの存在確認
            if not ai_file.exists():
                print(f"⚠️  AI file not found: {ai_file}")
                continue
            if not human_file.exists():
                print(f"⚠️  Human file not found: {human_file}")
                continue
            
            # 比較を実行
            try:
                comparator = XMLSectionComparison(str(ai_file), str(human_file))
                
                # データロードと抽出
                comparator.load_xml_files()
                ai_sections = comparator.extract_section_tree(comparator.ai_tree, "AI")
                human_sections = comparator.extract_section_tree(comparator.human_tree, "Human")
                
                # 統計計算
                ai_stats = comparator.calculate_tree_statistics(ai_sections, "AI")
                human_stats = comparator.calculate_tree_statistics(human_sections, "Human")
                
                # ページ単位の比較
                page_comparisons = comparator.align_sections_by_page(ai_sections, human_sections)
                
                # 全体メトリクス
                overall_metrics = comparator.calculate_overall_metrics(page_comparisons)
                
                # 結果を保存
                self.results.append({
                    'game_id': game_id,
                    'ai_stats': ai_stats,
                    'human_stats': human_stats,
                    'page_comparisons': page_comparisons,
                    'overall_metrics': overall_metrics
                })
                
                # 個別レポートを表示
                report = comparator.generate_report(ai_stats, human_stats, 
                                                   page_comparisons, overall_metrics)
                print("\n" + report)
                
                # 個別レポートをファイルに保存
                output_file = f"comparison_report_{game_id}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"\n💾 Individual report saved to: {output_file}")
                
            except Exception as e:
                print(f"❌ Error processing {game_id}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # 統合レポートを生成
        if self.results:
            self.generate_summary_report()
    
    def generate_summary_report(self):
        """全ゲームの統合レポートを生成"""
        print("\n" + "=" * 80)
        print("SUMMARY REPORT - All Games")
        print("=" * 80)
        
        report = []
        report.append("=" * 80)
        report.append("BATCH COMPARISON SUMMARY REPORT")
        report.append(f"Total games analyzed: {len(self.results)}")
        report.append("=" * 80)
        
        # 各ゲームのサマリー
        report.append("\n📊 PER-GAME SUMMARY:")
        report.append(f"{'Game ID':<20} {'AI Secs':<10} {'Hum Secs':<10} {'F1-Score':<10} {'Edit Dist':<12}")
        report.append("-" * 80)
        
        total_f1 = 0
        total_edit_dist = 0
        
        for result in self.results:
            game_id = result['game_id']
            ai_total = result['ai_stats']['total_sections']
            human_total = result['human_stats']['total_sections']
            metrics = result['overall_metrics']
            
            # F1スコアを計算
            tp = metrics['total_exact_matches']
            ai_types = metrics['total_ai_types']
            human_types = metrics['total_human_types']
            
            precision = tp / ai_types if ai_types > 0 else 0
            recall = tp / human_types if human_types > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            norm_edit_dist = metrics['normalized_edit_distance']
            
            total_f1 += f1_score
            total_edit_dist += norm_edit_dist
            
            report.append(
                f"{game_id:<20} "
                f"{ai_total:<10} "
                f"{human_total:<10} "
                f"{f1_score:<10.3f} "
                f"{norm_edit_dist:<12.3f}"
            )
        
        # 平均値
        avg_f1 = total_f1 / len(self.results)
        avg_edit_dist = total_edit_dist / len(self.results)
        
        report.append("-" * 80)
        report.append(f"{'AVERAGE':<20} {'':<10} {'':<10} {avg_f1:<10.3f} {avg_edit_dist:<12.3f}")
        
        # セクションタイプの集計
        report.append("\n📋 AGGREGATED SECTION TYPE STATISTICS:")
        
        # 全ゲームのタイプを集計
        all_ai_types = {}
        all_human_types = {}
        
        for result in self.results:
            for stype, count in result['ai_stats']['type_distribution'].items():
                all_ai_types[stype] = all_ai_types.get(stype, 0) + count
            for stype, count in result['human_stats']['type_distribution'].items():
                all_human_types[stype] = all_human_types.get(stype, 0) + count
        
        all_types = sorted(set(all_ai_types.keys()) | set(all_human_types.keys()))
        
        report.append(f"{'Type':<25} {'Total AI':<12} {'Total Human':<12} {'Diff':<10}")
        report.append("-" * 60)
        
        for stype in all_types:
            ai_count = all_ai_types.get(stype, 0)
            human_count = all_human_types.get(stype, 0)
            diff = ai_count - human_count
            diff_str = f"{diff:+d}" if diff != 0 else "0"
            report.append(f"{stype:<25} {ai_count:<12} {human_count:<12} {diff_str:<10}")
        
        # 総評
        report.append("\n💡 OVERALL ASSESSMENT:")
        if avg_f1 >= 0.8:
            report.append("  ✅ 優秀: AI生成のセクション構造は全体的に非常に高品質")
        elif avg_f1 >= 0.6:
            report.append("  ⚠️  良好: AI生成のセクション構造は概ね適切")
        elif avg_f1 >= 0.4:
            report.append("  ⚠️  要改善: セクション構造の精度向上が推奨される")
        else:
            report.append("  ❌ 不十分: セクション構造の大幅な改善が必要")
        
        report.append(f"\n平均F1スコア: {avg_f1:.3f}")
        report.append(f"平均正規化編集距離: {avg_edit_dist:.3f}")
        
        summary_text = "\n".join(report)
        print("\n" + summary_text)
        
        # サマリーレポートをファイルに保存
        with open("comparison_summary_report.txt", 'w', encoding='utf-8') as f:
            f.write(summary_text)
        print(f"\n💾 Summary report saved to: comparison_summary_report.txt")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="XML Batch Comparison Tool - Compare multiple game manuals"
    )
    parser.add_argument(
        "--ai-dir", 
        default="cursor_labeled_integrated",
        help="AI生成XMLファイルのディレクトリ（デフォルト: cursor_labeled_integrated）"
    )
    parser.add_argument(
        "--human-dir",
        default="output_images_xml_golden",
        help="人手作成XMLファイルのディレクトリ（デフォルト: output_images_xml_golden）"
    )
    parser.add_argument(
        "--game-ids",
        nargs='+',
        default=None,
        help="比較するゲームIDのリスト（省略時は自動検出）"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="human_dirから_human.xmlファイルを自動検出（デフォルト動作）"
    )
    
    args = parser.parse_args()
    
    # バッチ比較を実行（game_idsがNoneの場合は自動検出）
    batch_comparator = XMLBatchComparison(args.ai_dir, args.human_dir, args.game_ids)
    batch_comparator.run_batch_comparison()


if __name__ == "__main__":
    main()

