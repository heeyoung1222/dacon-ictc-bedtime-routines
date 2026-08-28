import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("Generating Tables for IEEE Paper...")

try:
    train_df = pd.read_csv('feature_table_train.csv')
    results_df = pd.read_csv('results_cv.csv')
    imp_df = pd.read_csv('feature_importance.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure previous step files exist.")
    exit()

# 1. Table 1: Dataset Characteristics (table_dataset.csv)
targets = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
dataset_stats = []

for t in targets:
    if t in train_df.columns:
        total = train_df[t].dropna().shape[0]
        positives = train_df[t].sum()
        ratio = (positives / total) * 100 if total > 0 else 0
        dataset_stats.append({
            'Target Label': t,
            'Total Instances': total,
            'Positive Class (1)': int(positives),
            'Positive Ratio (%)': f"{ratio:.1f}"
        })

df_dataset = pd.DataFrame(dataset_stats)
df_dataset.to_csv('table_dataset.csv', index=False)
print("Generated table_dataset.csv")

# 2. Table 2: Predictive Performance (table_result.csv)
res_table = results_df[['Label', 'LogLoss', 'Accuracy', 'Macro F1']].copy()
res_table['LogLoss'] = res_table['LogLoss'].apply(lambda x: f"{x:.3f}")
res_table['Accuracy'] = res_table['Accuracy'].apply(lambda x: f"{x:.3f}")
res_table['Macro F1'] = res_table['Macro F1'].apply(lambda x: f"{x:.3f}")

res_table.to_csv('table_result.csv', index=False)
print("Generated table_result.csv")

# 3. Table 3: Top Features by Context Category (table_feature.csv)
def categorize_feature(feat):
    if 'app' in feat: return 'App Usage'
    if 'wifi' in feat or 'ble' in feat: return 'Network Context'
    if 'gps' in feat: return 'Location Context'
    return 'Other Sensors'

imp_df['Category'] = imp_df['Feature'].apply(categorize_feature)

global_imp = imp_df.groupby(['Category', 'Feature'])['Importance'].mean().reset_index()
global_imp = global_imp.sort_values(['Category', 'Importance'], ascending=[True, False])

top_features_per_cat = global_imp.groupby('Category').head(3).reset_index(drop=True)
top_features_per_cat['Importance'] = top_features_per_cat['Importance'].apply(lambda x: f"{x:.4f}")

top_features_per_cat.to_csv('table_feature.csv', index=False)
print("Generated table_feature.csv")

print("\nAll required tables generated successfully.")