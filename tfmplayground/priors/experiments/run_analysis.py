"""CLI runner for regression data analysis."""

import os
from pathlib import Path
import sys
from typing import Dict, List
from itertools import combinations
from utils import load_config, discover_h5_files
from regression.analyzer import RegressionDataAnalyzer, compare_regression_priors
from classification.analyzer import ClassificationDataAnalyzer, compare_classification_priors


def select_priors_for_analysis(available_files: Dict[str, str]) -> List[str]:
    """Interactive selection of priors to analyze."""
    if not available_files:
        print("No data files found!")
        return []
    
    print("\n" + "=" * 50)
    print("AVAILABLE DATA FILES")
    print("=" * 50)
    
    prior_list = list(available_files.keys())
    for i, (name, path) in enumerate(available_files.items(), 1):
        print(f"{i}. {name:<15} - {os.path.basename(path)}")
    
    print("\n" + "=" * 50)
    print("SELECT PRIORS TO ANALYZE")
    print("=" * 50)
    print("Options:")
    print("  - Enter numbers separated by commas (e.g., '1,2')")
    print("  - Enter 'all' to analyze all priors")
    print("  - Enter 'quit' to exit")
    
    while True:
        selection = input("\nYour selection: ").strip().lower()
        
        if selection == 'quit':
            return []
        
        if selection == 'all':
            return prior_list
        
        try:
            indices = [int(x.strip()) for x in selection.split(',')]
            selected = [prior_list[i - 1] for i in indices if 1 <= i <= len(prior_list)]
            
            if not selected:
                print(f"Invalid selection. Please enter numbers between 1 and {len(prior_list)}.")
                continue
            
            print("\nSelected:", ", ".join(selected))
            confirm = input("Proceed? (y/n): ").strip().lower()
            if confirm == 'y':
                return selected
            
        except (ValueError, IndexError):
            print(f"Invalid input. Please enter numbers between 1 and {len(prior_list)} or 'all'.")


def select_analysis_mode() -> str:
    """Interactive selection of analysis mode."""
    print("\n" + "=" * 50)
    print("SELECT ANALYSIS MODE")
    print("=" * 50)
    print("1. Individual reports only")
    print("2. Comparisons only")
    print("3. Both individual reports and comparisons")
    
    while True:
        selection = input("\nYour choice (1-3): ").strip()
        
        if selection == '1':
            return 'individual'
        elif selection == '2':
            return 'comparison'
        elif selection == '3':
            return 'both'
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


def main():
    """Run interactive analysis CLI."""
    
    mode = sys.argv[1]  # Mode: classification, regression
    
    config = load_config(str(Path(__file__).parent / "config.yaml"))
    
    data_dir = Path(mode) /config['output']['data_dir']
    reports_dir = Path(mode) /config['output']['reports_dir']
    save_reports = config['output']['save_reports']
    
    os.makedirs(reports_dir, exist_ok=True)
    
    print("\n" + "=" * 50)
    print(f"{mode.upper()} DATA ANALYSIS")
    print("=" * 50)
    
    available_files = discover_h5_files(data_dir)
    
    if not available_files:
        print(f"\nNo data files found in {data_dir}")
        print("Please run data generation first (data_generation.py)")
        return
    
    selected_priors = select_priors_for_analysis(available_files)
    
    if not selected_priors:
        print("No priors selected. Exiting...")
        return
    
    analysis_mode = select_analysis_mode()
        
    print("\n" + "=" * 50)
    print("ANALYZING PRIORS")
    print("=" * 50)
    
    analyzers = {}
    
    # individual analysis
    if analysis_mode in ['individual', 'both']:
        for prior_name in selected_priors:
            file_path = available_files[prior_name]
            print(f"\n{'=' * 50}")
            print(f"Analyzing {prior_name.upper()}")
            print(f"{'=' * 50}")
            
            try:
                if mode == 'regression':
                    analyzer = RegressionDataAnalyzer(file_path)
                else:
                    analyzer = ClassificationDataAnalyzer(file_path)
                analyzers[prior_name] = analyzer
                
                report = analyzer.generate_report()
                
                if save_reports:
                    report_path = os.path.join(reports_dir, f"{prior_name}_analysis_report.txt")
                    with open(report_path, "w") as f:
                        f.write(report)
                    print(f"\n[OK] Report saved to {report_path}")
                
            except Exception as e:
                print(f"Error analyzing {prior_name}: {e}")
                continue
    else:
        # load analyzers without generating reports
        for prior_name in selected_priors:
            file_path = available_files[prior_name]
            try:
                if mode == 'regression':
                    analyzer = RegressionDataAnalyzer(file_path)
                else:
                    analyzer = ClassificationDataAnalyzer(file_path)
                analyzers[prior_name] = analyzer
            except Exception as e:
                print(f"Error loading {prior_name}: {e}")
                continue
    
    # pairwise comparisons
    if analysis_mode in ['comparison', 'both'] and len(analyzers) >= 2:
        print("\n" + "=" * 50)
        print("PAIRWISE COMPARISONS")
        
        for (name1, analyzer1), (name2, analyzer2) in combinations(analyzers.items(), 2):
            print(f"{'=' * 50}")
            print(f"Comparing {name1.upper()} vs {name2.upper()}")
            print(f"{'=' * 50}")
            
            try:
                if mode == 'regression':
                    comparison = compare_regression_priors(analyzer1, analyzer2, name1.upper(), name2.upper())
                else:
                    comparison = compare_classification_priors(analyzer1, analyzer2, name1.upper(), name2.upper())

                if save_reports:
                    comp_filename = f"comparison_{name1}_vs_{name2}.txt"
                    comp_path = os.path.join(reports_dir, comp_filename)
                    with open(comp_path, "w") as f:
                        f.write(comparison)
                    print(f"\n[OK] Comparison saved to {comp_path}")
                    
            except Exception as e:
                print(f"Error comparing {name1} vs {name2}: {e}")
    
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    if save_reports:
        print(f"Reports saved to: {reports_dir}")


if __name__ == "__main__":
    main()
