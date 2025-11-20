"""
===============================================================================
                    STEP 6: FEATURE ENGINEERING
                    
                    Создание новых признаков и финальная подготовка
===============================================================================

ЦЕЛЬ:
-----
1. Удалить дубли (person_income оставляем только log версию)
2. Создать новые признаки для субпрайм-логики
3. One-Hot Encoding для категорий
4. Масштабирование всех числовых
5. Запустить baseline v2 и измерить улучшение

ОЖИДАЕМЫЙ РЕЗУЛЬТАТ:
-------------------
- 20-25 признаков вместо 11
- ROC-AUC улучшение до 0.89-0.90
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import pickle

warnings.filterwarnings('ignore')

# Пути
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_PROCESSED = ROOT_DIR / 'data' / 'processed'
DATA_ENGINEERED = ROOT_DIR / 'data' / 'engineered'
RESULTS = ROOT_DIR / 'results' / 'step6_engineering'
MODEL_V2 = ROOT_DIR / 'results' / 'model_versions' / 'v2_after_engineering'

# Создаем папки
DATA_ENGINEERED.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
MODEL_V2.mkdir(parents=True, exist_ok=True)

def load_data():
    """Загрузка очищенных данных."""
    print("\n" + "="*60)
    print("STEP 6: FEATURE ENGINEERING")
    print("="*60)
    
    train_df = pd.read_csv(DATA_PROCESSED / 'train_cleaned.csv')
    test_df = pd.read_csv(DATA_PROCESSED / 'test_cleaned.csv')
    
    print(f"\n📊 Загружено:")
    print(f"   Train: {len(train_df)} записей, {len(train_df.columns)} признаков")
    print(f"   Test: {len(test_df)} записей")
    
    return train_df, test_df

def remove_duplicates(df):
    """Удаление дублированных признаков."""
    print("\n🔧 Удаление дублей:")
    print("-"*40)
    
    # Удаляем оригинальный person_income (оставляем только log версию)
    if 'person_income' in df.columns and 'person_income_log' in df.columns:
        df = df.drop('person_income', axis=1)
        print("   ✅ Удален person_income (оставлен person_income_log)")
    
    return df

def create_subprime_features(df):
    """Создание признаков специально для субпрайм-логики."""
    print("\n🚀 Создание НОВЫХ признаков:")
    print("-"*40)
    
    # 1. High Risk Score - комбинация плохих признаков
    df['high_risk_score'] = (
        (df['loan_grade'].isin(['F', 'G'])).astype(int) +
        (df['cb_person_default_on_file'] == 'Y').astype(int) +
        (df['person_home_ownership'] == 'RENT').astype(int) +
        (df['loan_percent_income'] > 0.25).astype(int)
    )
    print("   ✅ high_risk_score: сумма рисковых факторов (0-4)")
    
    # 2. Is Subprime Client - идеальный клиент для субпрайма
    df['is_subprime_client'] = (
        (df['loan_grade'].isin(['D', 'E', 'F', 'G'])) & 
        (df['loan_percent_income'] > 0.20)
    ).astype(int)
    print("   ✅ is_subprime_client: плохой грейд + высокая нагрузка")
    
    # 3. DTI категории (более детальные)
    df['dti_critical'] = (df['loan_percent_income'] > 0.35).astype(int)
    df['dti_high'] = (df['loan_percent_income'] > 0.25).astype(int)
    print("   ✅ dti_critical/high: критическая долговая нагрузка")
    
    # 4. Age группы для субпрайма
    df['age_risk_young'] = (df['person_age'] < 25).astype(int)
    df['age_risk_old'] = (df['person_age'] > 60).astype(int)
    print("   ✅ age_risk: молодые и пожилые")
    
    # 5. Income to loan ratio
    df['income_loan_ratio'] = df['person_income_log'] / (np.log1p(df['loan_amnt']) + 1)
    print("   ✅ income_loan_ratio: отношение дохода к кредиту")
    
    # 6. Bad grade with high rate
    df['bad_grade_high_rate'] = (
        (df['loan_grade'].isin(['E', 'F', 'G'])) & 
        (df['loan_int_rate'] > 15)
    ).astype(int)
    print("   ✅ bad_grade_high_rate: плохой грейд + высокая ставка")
    
    # 7. Консолидация долгов с плохим грейдом
    df['debt_consolidation_risk'] = (
        (df['loan_intent'] == 'DEBTCONSOLIDATION') & 
        (df['loan_grade'].isin(['D', 'E', 'F', 'G']))
    ).astype(int)
    print("   ✅ debt_consolidation_risk: консолидация + плохой грейд")
    
    # 8. Короткая кредитная история с высоким риском
    df['short_history_risk'] = (
        (df['cb_person_cred_hist_length'] < 3) & 
        (df['loan_grade'].isin(['F', 'G']))
    ).astype(int)
    print("   ✅ short_history_risk: мало истории + плохой грейд")
    
    print(f"\n   🎯 Создано 10 новых признаков!")
    
    return df

def encode_categorical(df, is_train=True):
    """One-Hot Encoding для категориальных признаков."""
    print("\n🔧 Кодирование категориальных:")
    print("-"*40)
    
    categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade']
    
    for col in categorical_cols:
        # One-hot encoding
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(col, axis=1)
        print(f"   ✅ {col}: создано {len(dummies.columns)} бинарных признаков")
    
    # cb_person_default_on_file - простое кодирование (Y=1, N=0)
    if 'cb_person_default_on_file' in df.columns:
        df['cb_person_default_on_file'] = (df['cb_person_default_on_file'] == 'Y').astype(int)
        print("   ✅ cb_person_default_on_file: Y→1, N→0")
    
    return df

def scale_features(train_df, test_df):
    """Масштабирование числовых признаков."""
    print("\n📏 Масштабирование признаков:")
    print("-"*40)
    
    # Определяем числовые столбцы (исключаем бинарные и таргет)
    numeric_cols = [
        'person_age', 'person_income_log', 'person_emp_length',
        'loan_amnt', 'loan_int_rate', 'loan_percent_income',
        'cb_person_cred_hist_length', 'income_loan_ratio'
    ]
    
    # Фильтруем только существующие
    numeric_cols = [col for col in numeric_cols if col in train_df.columns]
    
    # Создаем и обучаем StandardScaler
    scaler = StandardScaler()
    
    # Масштабируем train
    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    
    # Масштабируем test (используем параметры от train!)
    test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])
    
    print(f"   ✅ Масштабировано {len(numeric_cols)} числовых признаков")
    print(f"   Метод: StandardScaler (mean=0, std=1)")
    
    # Сохраняем scaler
    with open(DATA_ENGINEERED / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return train_df, test_df, scaler

def run_baseline_v2(train_df):
    """Запуск baseline на engineered данных."""
    print("\n" + "="*60)
    print("BASELINE MODEL v2 - After Feature Engineering")
    print("="*60)
    
    # Подготовка данных
    X = train_df.drop(['id', 'loan_status'], axis=1, errors='ignore')
    y = train_df['loan_status']
    
    print(f"\n📊 Размерность данных: {X.shape}")
    print(f"   Было признаков: 11")
    print(f"   Стало признаков: {X.shape[1]}")
    
    # Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # ========== LOGISTIC REGRESSION ==========
    print("\n🔵 Logistic Regression:")
    lr_model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    lr_model.fit(X_train, y_train)
    
    lr_pred_proba = lr_model.predict_proba(X_val)[:, 1]
    lr_auc = roc_auc_score(y_val, lr_pred_proba)
    print(f"   ROC-AUC: {lr_auc:.4f}")
    
    # ========== RANDOM FOREST ==========
    print("\n🌲 Random Forest:")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    rf_pred_proba = rf_model.predict_proba(X_val)[:, 1]
    rf_auc = roc_auc_score(y_val, rf_pred_proba)
    print(f"   ROC-AUC: {rf_auc:.4f}")
    
    # ========== СРАВНЕНИЕ ==========
    print("\n📊 СРАВНЕНИЕ С ПРЕДЫДУЩИМИ ВЕРСИЯМИ:")
    print("-"*50)
    
    baseline_v0 = 0.8740  # Исходный baseline
    baseline_v1 = 0.8730  # После очистки
    
    print(f"   v0_baseline (сырые):     {baseline_v0:.4f}")
    print(f"   v1_cleaned:              {baseline_v1:.4f}")
    print(f"   v2_LR (engineered):      {lr_auc:.4f} ({lr_auc-baseline_v0:+.4f})")
    print(f"   v2_RF (engineered):      {rf_auc:.4f} ({rf_auc-baseline_v0:+.4f}) 🔥")
    
    # ========== ВАЖНОСТЬ НОВЫХ ПРИЗНАКОВ ==========
    print("\n🎯 Топ-10 признаков (Random Forest):")
    print("-"*40)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        bar = '█' * int(row['importance'] * 50)
        print(f"{row['feature']:30s}: {row['importance']:.3f} {bar}")
    
    # Проверяем попали ли новые признаки в топ
    new_features = ['high_risk_score', 'is_subprime_client', 'dti_critical', 
                   'bad_grade_high_rate', 'debt_consolidation_risk']
    new_in_top = feature_importance.head(15)['feature'].isin(new_features).sum()
    print(f"\n✨ Новых признаков в топ-15: {new_in_top} из {len(new_features)}")
    
    # ========== СОХРАНЕНИЕ ==========
    with open(MODEL_V2 / 'metrics.txt', 'w') as f:
        f.write("BASELINE MODEL v2 - After Feature Engineering\n")
        f.write("="*50 + "\n")
        f.write(f"Logistic Regression ROC-AUC: {lr_auc:.4f}\n")
        f.write(f"Random Forest ROC-AUC: {rf_auc:.4f}\n")
        f.write(f"Improvement from v0: {rf_auc-baseline_v0:+.4f}\n")
        f.write(f"\nTotal features: {X.shape[1]}\n")
        f.write("\nNew features created:\n")
        for feat in new_features:
            f.write(f"- {feat}\n")
    
    # Сохраняем лучшую модель
    best_model = rf_model if rf_auc > lr_auc else lr_model
    with open(MODEL_V2 / 'model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    return rf_auc, feature_importance

def main():
    """Основная функция."""
    
    # Загрузка
    train_df, test_df = load_data()
    
    # Удаление дублей
    train_df = remove_duplicates(train_df)
    test_df = remove_duplicates(test_df)
    
    # Создание новых признаков
    train_df = create_subprime_features(train_df)
    test_df = create_subprime_features(test_df)
    
    # Кодирование категориальных
    train_df = encode_categorical(train_df, is_train=True)
    test_df = encode_categorical(test_df, is_train=False)
    
    # Масштабирование
    train_df, test_df, scaler = scale_features(train_df, test_df)
    
    # Сохранение engineered данных
    train_df.to_csv(DATA_ENGINEERED / 'train_engineered.csv', index=False)
    test_df.to_csv(DATA_ENGINEERED / 'test_engineered.csv', index=False)
    print(f"\n💾 Engineered данные сохранены в {DATA_ENGINEERED}")
    
    # Запуск baseline v2
    rf_auc, feature_importance = run_baseline_v2(train_df)
    
    # Итоги
    print("\n" + "="*60)
    print("STEP 6 COMPLETED!")
    print("="*60)
    print(f"✅ Создано 10 новых признаков")
    print(f"✅ Всего признаков: {len(train_df.columns)-2}")
    print(f"✅ Random Forest ROC-AUC: {rf_auc:.4f}")
    print(f"✅ Результаты в {MODEL_V2}")

if __name__ == "__main__":
    main()