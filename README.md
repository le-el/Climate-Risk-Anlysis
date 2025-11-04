# Physical Risk & Resilience Framework Analysis Pipeline

This pipeline analyzes company documents against the Physical Risk Resilience Framework using GPT-4o-mini for structured data extraction and rubric-based scoring (0-5 scale).

## Pipeline Overview

1. **Phase 1**: Preprocess documents (PDFs/HTML) → text chunks
2. **Phase 2-4**: Extract structured data using GPT-4o-mini → Score using rubric rules

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API key:**
   - The `.env` file has been created with your OpenAI API key
   - Make sure it's properly configured

3. **Required files/folders:**
   - `PhysicalRisk_Resilience_Framework.xlsx` - Framework with measures and rubrics
   - `Collected_Data/` - Folder with company data (PDFs and HTML)

## Usage

### Option 1: Run Full Pipeline (Recommended)
```bash
python run_full_pipeline.py
```

This will:
1. Preprocess all documents into chunks
2. Extract structured data for each measure using GPT-4o-mini
3. Score each measure using rubric rules (0-5 scale)
4. Generate Excel and JSON reports

### Option 2: Run Phases Separately

**Phase 1 only (Preprocessing):**
```bash
python phase1_preprocess.py
```

**Phase 2-4 (Extraction & Scoring):**
```bash
python phase2_extract_and_score.py
```

## Output

- `preprocessed_chunks/` - Text chunks from documents (one JSON file per company)
- `physical_risk_analysis_report.json` - Full results with extracted fields
- `physical_risk_analysis_report.xlsx` - Excel report with:
  - Detailed_Results: All measures with scores
  - Summary_by_Category: Scores grouped by category
  - Overall_Summary: Company-level summaries

## How It Works

For each company and measure:

1. **Keyword Search**: Find relevant text chunks using framework keywords
2. **LLM Extraction**: Send top chunks to GPT-4o-mini with JSON schema
3. **Data Extraction**: Get structured data (e.g., committee name, frequency, coverage %)
4. **Rubric Scoring**: Apply Python functions that encode rubric rules (0-5 scale)

## Scoring

Scores follow the framework rubric (0-5 scale):
- **0**: No mention/evidence
- **1**: Basic mention
- **2**: Some structure
- **3**: Explicit with basic requirements met
- **4**: Comprehensive coverage
- **5**: Enterprise-wide with assurance

## Notes

- Each measure requires custom JSON schema and scoring function
- Currently implemented: "Board-level physical risk oversight"
- Other measures use generic fallback scoring
- Can be extended with measure-specific schemas and scoring functions

## Cost Estimation

- GPT-4o-mini: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens
- Typical run: ~1-5 API calls per measure per company
- For 44 measures × 1 company: ~44-220 API calls
