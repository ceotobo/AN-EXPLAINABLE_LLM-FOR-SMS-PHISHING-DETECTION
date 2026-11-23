# evaluate_baselines.py -- Combine all baseline results into one summary table
# Run with     : python src/evaluate_baselines.py
# Prerequisite : All four baseline scripts must have run successfully first.

import pandas as pd
import os

# ── 1. Collect all result files ───────────────────────────────────────────────
files = {
    'Logistic Regression': 'outputs/lr_results.csv',
    'Random Forest'      : 'outputs/rf_results.csv',
    'XGBoost'            : 'outputs/xgb_results.csv',
    'DistilBERT'         : 'outputs/distilbert_results.csv',
}

all_results = []
missing     = []

for model_name, path in files.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        all_results.append(df)
        print(f'  Loaded: {path}')
    else:
        missing.append(model_name)
        print(f'  WARNING: {path} not found -- run that baseline script first.')

if not all_results:
    print('\nNo results found. Run all four baseline scripts first.')
    exit(1)

# ── 2. Combine and format ─────────────────────────────────────────────────────
combined = pd.concat(all_results, ignore_index=True)

# Select and rename for dissertation Table 4.1
summary = combined[['model', 'accuracy', 'precision', 'recall', 'f1', 'fpr']].copy()
summary.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'FPR']

for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'FPR']:
    summary[col] = summary[col].round(4)

# ── 3. Print table ────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('BASELINE RESULTS SUMMARY  --  Table 4.1 (Dissertation)')
print('Dataset: Dataset_5971.csv | Split: Validation set (15%)')
print('='*70)
print(summary.to_string(index=False))
print('='*70)

# ── 4. Best model indicator ───────────────────────────────────────────────────
best_f1  = summary.loc[summary['F1-Score'].idxmax(), 'Model']
best_fpr = summary.loc[summary['FPR'].idxmin(), 'Model']
print(f'\nHighest F1-Score : {best_f1}')
print(f'Lowest FPR       : {best_fpr}')
print('\nNote: FPR (False Positive Rate) = ham messages wrongly flagged as malicious.')
print('      A low FPR is critical -- blocking legitimate messages harms users.')

# ── 5. Confusion matrix breakdown ────────────────────────────────────────────
if all(c in combined.columns for c in ['tn', 'fp', 'fn', 'tp']):
    print('\nConfusion Matrix Breakdown:')
    cm_table = combined[['model', 'tn', 'fp', 'fn', 'tp']].copy()
    cm_table.columns = ['Model', 'TN (correct ham)', 'FP (ham→malicious)',
                        'FN (missed malicious)', 'TP (correct malicious)']
    print(cm_table.to_string(index=False))

# ── 6. Save combined results ──────────────────────────────────────────────────
os.makedirs('outputs', exist_ok=True)
summary.to_csv('outputs/baseline_results_all.csv', index=False)
combined.to_csv('outputs/baseline_results_full.csv', index=False)

print('\nSaved:')
print('  outputs/baseline_results_all.csv  -- Table 4.1 (copy into dissertation)')
print('  outputs/baseline_results_full.csv -- full version with confusion matrix counts')

if missing:
    print(f'\nStill missing results for: {", ".join(missing)}')
    print('Run those scripts to complete Table 4.1.')
else:
    print('\nAll four baselines complete. Table 4.1 is ready.')
    print('Next step: proceed to LLM agent development (src/llm_agent.py)')
