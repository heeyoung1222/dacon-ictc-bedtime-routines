import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

print("Generating Clean Figures without inside titles...")

try:
    results_df = pd.read_csv('results_cv.csv')
    imp_df = pd.read_csv('feature_importance.csv')
    train_df = pd.read_csv('feature_table_train.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure previous step files exist.")
    exit()

# 1. Figure 1: Proposed Framework
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.axis('off')
blocks = [
    (0.05, 0.3, 0.2, 0.4, 'Multimodal\nLifelogs\n(Sensors & Apps)'),
    (0.35, 0.3, 0.25, 0.4, 'Contextual\nFeature Extraction\n(Network, Location, App)'),
    (0.7, 0.3, 0.25, 0.4, 'Tree-based\nPrediction Model\n(Stress & Sleep)')
]
for x, y, w, h, text in blocks:
    rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='black', facecolor='#f0f0f0')
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, weight='bold')
ax.annotate('', xy=(0.35, 0.5), xytext=(0.25, 0.5), arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('', xy=(0.7, 0.5), xytext=(0.6, 0.5), arrowprops=dict(facecolor='black', shrink=0.05))
plt.tight_layout()
plt.savefig('figure1_framework.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Figure 2: Aggregated Feature Importance by Context
def categorize_feature(feat):
    if 'app' in feat: return 'App Usage'
    if 'wifi' in feat or 'ble' in feat: return 'Network Context'
    if 'gps' in feat: return 'Location Context'
    return 'Other Sensors'

imp_df['Category'] = imp_df['Feature'].apply(categorize_feature)
cat_imp = imp_df.groupby('Category')['Importance'].sum().reset_index()
cat_imp = cat_imp.sort_values('Importance', ascending=False)

plt.figure(figsize=(6, 3.5))
sns.barplot(x='Importance', y='Category', data=cat_imp, palette='Greys_r')
plt.xlabel('Cumulative Importance Score')
plt.ylabel('Feature Category')
plt.tight_layout()
plt.savefig('figure2_dbrw.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Figure 3: Predictive Performance
fig, ax1 = plt.subplots(figsize=(7, 3.5))
x = np.arange(len(results_df['Label']))
width = 0.35
ax1.bar(x - width/2, results_df['LogLoss'], width, label='Log Loss', color='#555555')
ax1.set_ylabel('Log Loss')
ax1.set_xlabel('Target Labels')
ax1.set_xticks(x)
ax1.set_xticklabels(results_df['Label'])
ax1.set_ylim(0, 1.0)

ax2 = ax1.twinx()
ax2.bar(x + width/2, results_df['Macro F1'], width, label='Macro F1', color='#aaaaaa')
ax2.set_ylabel('Macro F1 Score')
ax2.set_ylim(0, 1.0)

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')
plt.tight_layout()
plt.savefig('figure3_window_result.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Figure 4: Global Top 10 Features
global_imp = imp_df.groupby('Feature')['Importance'].mean().reset_index()
global_imp = global_imp.sort_values('Importance', ascending=False).head(10)

plt.figure(figsize=(8, 4.5))
sns.barplot(x='Importance', y='Feature', data=global_imp, palette='binary_r')
plt.xlabel('Mean Importance')
plt.ylabel('Feature Name')
plt.tight_layout()
plt.savefig('figure4_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Figure 5: Personalized Example
plt.figure(figsize=(5, 3.5))
target_feature = 'w18_24_app_top_캐시워크'
if target_feature in train_df.columns:
    sns.boxplot(x='S2', y=target_feature, data=train_df, palette='Greys')
    plt.xlabel('Stress Label (S2)')
    plt.ylabel('Cashwalk App Usage (w18_24)')
    plt.tight_layout()
    plt.savefig('figure5_personalized_example.png', dpi=300, bbox_inches='tight')
plt.close()

print("Clean figures generated successfully without embedded titles.")