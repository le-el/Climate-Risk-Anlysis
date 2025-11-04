"""
Complete Pipeline Runner
Runs Phase 1 (preprocessing) and Phase 2-4 (extraction and scoring)
"""

import os
import sys

def main():
    print("="*80)
    print("Physical Risk & Resilience Framework Analysis Pipeline")
    print("="*80)
    
    # Phase 1: Preprocessing
    print("\n[PHASE 1] Preprocessing documents...")
    print("-" * 80)
    from phase1_preprocess import preprocess_company_data
    
    if not os.path.exists("Collected_Data"):
        print("Error: Collected_Data folder not found!")
        sys.exit(1)
    
    preprocess_company_data("Collected_Data", output_folder="preprocessed_chunks")
    
    # Phase 2-4: Extraction and Scoring
    print("\n\n[PHASE 2-4] Extracting data and scoring...") 
    print("-" * 80)
    from phase2_extract_and_score import run_full_analysis
    
    if not os.path.exists("preprocessed_chunks"):
        print("Error: Preprocessing failed or chunks not found!")
        sys.exit(1)
    
    if not os.path.exists("PhysicalRisk_Resilience_Framework.xlsx"):
        print("Error: Framework file not found!")
        sys.exit(1)
    
    run_full_analysis(
        framework_path='PhysicalRisk_Resilience_Framework.xlsx',
        chunks_folder='preprocessed_chunks',
        output_file=None  # None means save one file per company in Result folder
    )
    
    print("\n" + "="*80)
    print("✓ Pipeline complete!")
    print("="*80)

if __name__ == "__main__":
    main()
