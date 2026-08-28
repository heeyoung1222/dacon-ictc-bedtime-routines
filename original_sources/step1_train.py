import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss, accuracy_score, f1_score
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# 1. 설정 및 데이터 로드
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

print("Loading data...")
df = pd.read_csv('feature_table_train.csv')
df['sleep_date'] = pd.to_datetime(df['sleep_date'])

# personal_ feature 제거
personal_cols = [c for c in df.columns if c.startswith('personal_')]
df = df.drop(columns=personal_cols)

results = []

def calibrate_probs(P, prior, alpha):
    return alpha * P + (1 - alpha) * prior

for target in TARGETS:
    print(f"\\n--- Training for {target} ---")
    
    # 2. Label별 Feature 필터링
    target_window = WINDOW_MAP[target]
    
    drop_targets = [t for t in TARGETS if t != target]
    df_target = df.drop(columns=drop_targets).dropna(subset=[target])
    
    # Window 기반 컬럼 필터링
    # 'all'이 아닌 경우, 다른 window 컬럼은 제외 (단, 기초 정보나 day_ 등은 기준에 따라 처리)
    # 여기서는 명확성을 위해 특정 window에 해당하는 컬럼만 남기거나, window 정보가 없는 공통 컬럼만 남김
    use_cols = []
    for c in df_target.columns:
        if c in ['subject_id', 'sleep_date', 'lifelog_date', 'split', target]:
            use_cols.append(c)
        elif target_window == 'all':
            use_cols.append(c)
        else:
            # 특정 window 문자열이 포함되어 있거나, 아예 window 표시(_w, _day)가 없는 경우 보존
            if target_window in c or not any(w in c for w in ['_day_', '_w18_24_', '_w20_24_', '_w21_24_', '_w22_24_', '_w20_02_']):
                use_cols.append(c)
                
    df_filtered = df_target[use_cols]
    
    # 3. Subject-wise Time-based Split (80% Train, 20% Val)
    train_indices = []
    val_indices = []
    
    for subj, group in df_filtered.groupby('subject_id'):
        group = group.sort_values('sleep_date')
        n_train = int(len(group) * 0.8)
        
        # 데이터가 너무 적은 경우 예외 처리
        if n_train == 0 or n_train == len(group):
            continue
            
        train_indices.extend(group.iloc[:n_train].index)
        val_indices.extend(group.iloc[n_train:].index)
        
    train_df = df_filtered.loc[train_indices]
    val_df = df_filtered.loc[val_indices]
    
    X_train = train_df.drop(columns=['subject_id', 'sleep_date', 'lifelog_date', 'split', target], errors='ignore')
    y_train = train_df[target].astype(int)
    
    X_val = val_df.drop(columns=['subject_id', 'sleep_date', 'lifelog_date', 'split', target], errors='ignore')
    y_val = val_df[target].astype(int)
    
    # 4. 결측치 처리 (간단히 0으로 채움. 필요시 이전 파이프라인 수식 적용)
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    
    # 5. 모델 학습 (OOB Score를 위해 bootstrap=True 설정)
    clf = ExtraTreesClassifier(n_estimators=200, bootstrap=True, oob_score=True, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # 6. Custom Calibration 적용
    prior_train = y_train.mean()
    oob_probs = clf.oob_decision_function_[:, 1]
    
    # 최적의 alpha 찾기 (Train OOB 데이터 기준 LogLoss 최소화)
    def loss_for_alpha(a):
        calibrated = calibrate_probs(oob_probs, prior_train, a)
        # 범위를 1e-15 ~ 1-1e-15로 제한하여 log_loss 에러 방지
        calibrated = np.clip(calibrated, 1e-15, 1 - 1e-15)
        return log_loss(y_train, calibrated)
        
    res = minimize_scalar(loss_for_alpha, bounds=(0, 1), method='bounded')
    best_alpha = res.x
    print(f"Optimal Alpha for {target}: {best_alpha:.4f} (Prior: {prior_train:.4f})")
    
    # 검증 데이터에 평가
    val_probs_raw = clf.predict_proba(X_val)[:, 1]
    val_probs_calibrated = calibrate_probs(val_probs_raw, prior_train, best_alpha)
    val_probs_calibrated = np.clip(val_probs_calibrated, 1e-15, 1 - 1e-15)
    
    val_preds = (val_probs_calibrated > 0.5).astype(int)
    
    ll = log_loss(y_val, val_probs_calibrated)
    acc = accuracy_score(y_val, val_preds)
    f1 = f1_score(y_val, val_preds, average='macro')
    
    print(f"Validation LogLoss: {ll:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")
    
    results.append({
        'Label': target,
        'LogLoss': ll,
        'Accuracy': acc,
        'Macro F1': f1,
        'Best Alpha': best_alpha
    })

# 7. 결과 저장
results_df = pd.DataFrame(results)
results_df.to_csv('results_cv.csv', index=False)
print("\\nSaved results to results_cv.csv")
print(results_df)