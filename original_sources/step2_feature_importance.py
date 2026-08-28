import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
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

print("Loading data for Feature Importance...")
df = pd.read_csv('feature_table_train.csv')
df['sleep_date'] = pd.to_datetime(df['sleep_date'])

# 정보 누출 방지를 위한 personal 피처 제거
personal_cols = [c for c in df.columns if c.startswith('personal_')]
df = df.drop(columns=personal_cols)

importance_list = []

for target in TARGETS:
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
    
    # Feature Importance는 전체 데이터를 활용하여 안정적으로 산출
    X = df_filtered.drop(columns=['subject_id', 'sleep_date', 'lifelog_date', 'split', target], errors='ignore')
    y = df_filtered[target].astype(int)
    
    X = X.fillna(0)
    
    # OOB는 불필요하므로 제거하고 최대한 트리를 키워 안정성 확보
    clf = ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    
    importances = clf.feature_importances_
    
    for col, imp in zip(X.columns, importances):
        importance_list.append({
            'Label': target,
            'Feature': col,
            'Importance': imp
        })

# DataFrame 변환 및 정렬
imp_df = pd.DataFrame(importance_list)

# 각 Label별 Top 30 추출
top_features = imp_df.sort_values(['Label', 'Importance'], ascending=[True, False])
top_features = top_features.groupby('Label').head(30).reset_index(drop=True)

# 파일 저장
top_features.to_csv('feature_importance.csv', index=False)
print("Saved Top 30 Feature Importances to feature_importance.csv")

# 빠른 확인을 위한 출력
for target in TARGETS:
    print(f"\nTop 3 features for {target}:")
    top3 = top_features[top_features['Label'] == target].head(3)
    for _, row in top3.iterrows():
        print(f"  - {row['Feature']}: {row['Importance']:.4f}")