import pandas as pd
import json

# Load the CSV
df = pd.read_csv('COMPREHENSIVE_DETAILED_ASSESSMENT_ALL_COMPANIES.csv')

print("="*80)
print("COMPREHENSIVE CSV FILE ANALYSIS")
print("="*80)

print(f"\n1. BASIC STATISTICS")
print(f"   Total Records: {len(df):,}")
print(f"   Total Columns: {len(df.columns)}")
print(f"   Unique Companies: {df['company_name'].nunique()}")
print(f"   Unique Measures: {df['measure_name'].nunique()}")

print(f"\n2. COMPANIES ANALYZED")
for company in df['company_name'].unique():
    count = len(df[df['company_name'] == company])
    avg_score = df[df['company_name'] == company]['score'].mean()
    print(f"   - {company}: {count} records, avg score: {avg_score:.2f}")

print(f"\n3. SCORE DISTRIBUTION (Overall)")
score_dist = df['score'].value_counts().sort_index()
for score, count in score_dist.items():
    pct = count / len(df) * 100
    print(f"   Score {score}: {count:4d} records ({pct:5.1f}%)")

print(f"\n4. SCORE STATISTICS")
print(f"   Average Score: {df['score'].mean():.2f}/5")
print(f"   Median Score: {df['score'].median():.2f}/5")
print(f"   Records with Score > 0: {(df['score'] > 0).sum():,} ({(df['score'] > 0).sum()/len(df)*100:.1f}%)")
print(f"   Records with Score = 0: {(df['score'] == 0).sum():,} ({(df['score'] == 0).sum()/len(df)*100:.1f}%)")

print(f"\n5. TOP MEASURES (by count)")
measure_counts = df['measure_name'].value_counts().head(10)
for measure, count in measure_counts.items():
    print(f"   - {measure}: {count} records")

print(f"\n6. FIELD POPULATION")
field_cols = [c for c in df.columns if c.startswith('field_')]
print(f"   Total field columns: {len(field_cols)}")
populated = [(col, df[col].notna().sum()) for col in field_cols]
populated.sort(key=lambda x: x[1], reverse=True)
print(f"   Top 10 most populated fields:")
for col, count in populated[:10]:
    print(f"     {col}: {count} non-null values")

print(f"\n7. SCORE BY COMPANY")
company_scores = df.groupby('company_name')['score'].agg(['mean', 'count', 'min', 'max']).sort_values('mean', ascending=False)
for company, row in company_scores.iterrows():
    print(f"   {company}:")
    print(f"     Avg: {row['mean']:.2f}, Count: {int(row['count'])}, Range: {int(row['min'])}-{int(row['max'])}")

print(f"\n8. CATEGORIES")
if 'category' in df.columns:
    cat_stats = df.groupby('category')['score'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    for category, row in cat_stats.iterrows():
        print(f"   {category}: avg {row['mean']:.2f}, {int(row['count'])} records")

print("\n" + "="*80)
