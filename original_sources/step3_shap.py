import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
WINDOW_MAP = {
    'Q1': 'all',
    'Q2': 'w18_24',
    'Q3': 'day',
    'S1': 'all',
    'S2': 'w21_24',
    'S3': 'all',
    'S4': 'w21_24'
}

print("Loading data for SHAP Analysis...")
df = pd.read_csv('feature_table_train.csv')
df['sleep_date'] = pd.to_datetime(df['sleep_date'])

# 정보 누출 및 시계열 과적합(Leakage) 방지
personal_cols = [c for c in df.columns if c.startswith('personal_')]
leak_cols = ['month'] # month 강제 제거
df = df.drop(columns=personal_cols + [c for c in leak_cols if c in df.columns])

for target in TARGETS:
    print(f"\nProcessing SHAP for {target}...")
    target_window = WINDOW_MAP[target]
    
    drop_targets = [t for t in TARGETS if t != target]
    df_target = df.drop(columns=drop_targets).dropna(subset=[target])
    
    use_cols = []
    for c in df_target.columns:
        if c in ['subject_id', 'sleep_date', 'lifelog_date', 'split', target]:
            use_cols.append(c)
        elif target_window == 'all':
            use_cols.append(c)
        else:
            if target_window in c or not any(w in c for w in ['_day_', '_w18_24_', '_w20_24_', '_w21_24_', '_w22_24_', '_w20_02_']):
                use_cols.append(c)
                
    df_filtered = df_target[use_cols]
    
    X = df_filtered.drop(columns=['subject_id', 'sleep_date', 'lifelog_date', 'split', target], errors='ignore')
    y = df_filtered[target].astype(int)
    X = X.fillna(0)
    
    # SHAP 분석을 위한 모델 재학습 (깊이를 제한하여 과적합 방지 및 설명력 확보)
    clf = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
    clf.fit(X, y)
    
    # SHAP 연산 시간 단축을 위해 500개 샘플링
    X_sample = shap.sample(X, 500, random_state=42) if len(X) > 500 else X
    
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)
    
    # ExtraTrees 이진 분류의 경우 shap_values가 리스트 형태로 반환될 수 있음
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
        
    # 1. SHAP Bar Plot (Feature Importance)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False, max_display=15)
    plt.title(f'SHAP Feature Importance - {target}')
    plt.tight_layout()
    plt.savefig(f'shap_bar_{target}.png', dpi=300)
    plt.close()
    
    # 2. SHAP Dot Plot (방향성 확인용)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type='dot', show=False, max_display=15)
    plt.title(f'SHAP Summary (Direction) - {target}')
    plt.tight_layout()
    plt.savefig(f'shap_dot_{target}.png', dpi=300)
    plt.close()

print("\nSaved all SHAP figures.")