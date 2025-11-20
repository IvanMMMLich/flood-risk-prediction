"""
===============================================================================
                    STEP 3: DATA QUALITY CHECK & CLEANING
                    
                    Проверка качества и очистка данных
===============================================================================

ЦЕЛЬ:
-----
1. Заменить аномальные значения (123)
2. Проверить пропуски и дубликаты
3. Обработать выбросы
4. Логарифмировать скошенные распределения
5. Запустить baseline v1 на очищенных данных

ОЖИДАЕМЫЙ РЕЗУЛЬТАТ:
-------------------
- Чистый датасет без аномалий
- Улучшение baseline на ~0.005-0.010
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import pickle

warnings.filterwarnings('ignore')

# Настройка путей
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_RAW = ROOT_DIR / 'data' / 'raw'
DATA_PROCESSED = ROOT_DIR / 'data' / 'processed'
RESULTS = ROOT_DIR / 'results' / 'step3_cleaning'
MODEL_V1 = ROOT_DIR / 'results' / 'model_versions' / 'v1_after_cleaning'

# Создаем папки
RESULTS.mkdir(parents=True, exist_ok=True)
MODEL_V1.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

def load_data():
    """Загрузка исходных данных."""
    print("\n" + "="*60)
    print("STEP 3: DATA QUALITY CHECK & CLEANING")
    print("="*60)
    
    train_df = pd.read_csv(DATA_RAW / 'train.csv')
    test_df = pd.read_csv(DATA_RAW / 'test.csv')
    
    print(f"\n📊 Загружено:")
    print(f"   Train: {len(train_df)} записей")
    print(f"   Test: {len(test_df)} записей")
    
    return train_df, test_df

def check_anomalies(df, df_name="DataFrame"):
    """Проверка аномальных значений."""
    print(f"\n🔍 Проверка аномалий в {df_name}:")
    print("-"*40)
    
    anomalies = {}
    
    # Проверяем 123 в возрасте
    age_123 = (df['person_age'] == 123).sum()
    if age_123 > 0:
        print(f"   ⚠️ person_age = 123: {age_123} записей")
        anomalies['person_age'] = age_123
    
    # Проверяем 123 в стаже
    emp_123 = (df['person_emp_length'] == 123).sum()
    if emp_123 > 0:
        print(f"   ⚠️ person_emp_length = 123: {emp_123} записей")
        anomalies['person_emp_length'] = emp_123
    
    if not anomalies:
        print("   ✅ Аномалий не найдено")
    
    return anomalies

def fix_anomalies(df):
    """Исправление аномальных значений."""
    print("\n🔧 Исправление аномалий:")
    print("-"*40)
    
    df_clean = df.copy()
    
    # Заменяем 123 на медиану для возраста
    if (df_clean['person_age'] == 123).any():
        median_age = df_clean[df_clean['person_age'] != 123]['person_age'].median()
        count_age = (df_clean['person_age'] == 123).sum()
        df_clean.loc[df_clean['person_age'] == 123, 'person_age'] = median_age
        print(f"   ✅ person_age: заменено {count_age} значений на медиану {median_age:.0f}")
    
    # Заменяем 123 на медиану для стажа
    if (df_clean['person_emp_length'] == 123).any():
        median_emp = df_clean[df_clean['person_emp_length'] != 123]['person_emp_length'].median()
        count_emp = (df_clean['person_emp_length'] == 123).sum()
        df_clean.loc[df_clean['person_emp_length'] == 123, 'person_emp_length'] = median_emp
        print(f"   ✅ person_emp_length: заменено {count_emp} значений на медиану {median_emp:.0f}")
    
    return df_clean

def check_missing_values(df):
    """Проверка пропущенных значений."""
    print("\n🔍 Проверка пропусков:")
    print("-"*40)
    
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    
    if missing.sum() == 0:
        print("   ✅ Пропусков не найдено")
    else:
        missing_df = pd.DataFrame({
            'Missing': missing[missing > 0],
            'Percent': missing_pct[missing > 0]
        }).sort_values('Percent', ascending=False)
        print(missing_df)
    
    return missing

def check_duplicates(df):
    """Проверка дубликатов."""
    print("\n🔍 Проверка дубликатов:")
    print("-"*40)
    
    # Полные дубликаты
    full_duplicates = df.duplicated().sum()
    print(f"   Полных дубликатов: {full_duplicates}")
    
    # Дубликаты по ID (если есть)
    if 'id' in df.columns:
        id_duplicates = df['id'].duplicated().sum()
        print(f"   Дубликатов по ID: {id_duplicates}")
    
    if full_duplicates == 0:
        print("   ✅ Дубликатов не найдено")
    
    return full_duplicates

def handle_skewness(df):
    """Обработка скошенных распределений."""
    print("\n📊 Обработка скошенности:")
    print("-"*40)
    
    df_clean = df.copy()
    
    # Логарифмируем person_income
    skew_before = df_clean['person_income'].skew()
    df_clean['person_income_log'] = np.log1p(df_clean['person_income'])
    skew_after = df_clean['person_income_log'].skew()
    
    print(f"   person_income skewness:")
    print(f"   До: {skew_before:.2f}")
    print(f"   После log: {skew_after:.2f} ✅")
    
    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # До
    axes[0].hist(df['person_income'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_title(f'Original Income (skew={skew_before:.2f})')
    axes[0].set_xlabel('Income')
    axes[0].set_ylabel('Frequency')
    
    # После
    axes[1].hist(df_clean['person_income_log'], bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_title(f'Log Income (skew={skew_after:.2f})')
    axes[1].set_xlabel('Log(Income+1)')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(RESULTS / 'income_transformation.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    return df_clean

def analyze_outliers(df):
    """Анализ выбросов."""
    print("\n📊 Анализ выбросов (IQR метод):")
    print("-"*40)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers_summary = {}
    
    for col in numeric_cols:
        if col not in ['id', 'loan_status']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outliers_pct = (outliers / len(df)) * 100
            
            if outliers > 0:
                outliers_summary[col] = {
                    'count': outliers,
                    'percent': outliers_pct,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
                print(f"   {col}: {outliers} ({outliers_pct:.1f}%)")
    
    return outliers_summary

def run_baseline_v1(train_df_clean):
    """Запуск baseline на очищенных данных."""
    print("\n" + "="*60)
    print("BASELINE MODEL v1 - After Cleaning")
    print("="*60)
    
    # Подготовка данных (аналогично baseline_model.py)
    X = train_df_clean.drop(['id', 'loan_status'], axis=1)
    y = train_df_clean['loan_status']
    
    # Кодирование категориальных
    le_dict = {}
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        X.loc[:, col] = le.fit_transform(X[col])
        le_dict[col] = le
    
    # Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Обучение
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Предсказания
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)
    
    # Метрики
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    print(f"\n📈 РЕЗУЛЬТАТЫ:")
    print(f"   ROC-AUC: {roc_auc:.4f}")
    
    # Сравнение с baseline
    baseline_auc = 0.8740  # Из v0
    improvement = roc_auc - baseline_auc
    print(f"\n📊 СРАВНЕНИЕ С BASELINE:")
    print(f"   v0_baseline: {baseline_auc:.4f}")
    print(f"   v1_cleaned:  {roc_auc:.4f}")
    print(f"   Улучшение:   {improvement:+.4f} {'✅' if improvement > 0 else '❌'}")
    
    # Сохранение метрик
    with open(MODEL_V1 / 'metrics.txt', 'w') as f:
        f.write("BASELINE MODEL v1 - After Cleaning\n")
        f.write("="*50 + "\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"Improvement from v0: {improvement:+.4f}\n")
        f.write("\nChanges applied:\n")
        f.write("- Replaced 123 anomalies with median\n")
        f.write("- Log transformation of person_income\n")
        f.write("- No outliers removed (kept for subprime logic)\n")
    
    # Сохранение модели
    with open(MODEL_V1 / 'model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return roc_auc, improvement

def main():
    """Основная функция."""
    
    # 1. Загрузка данных
    train_df, test_df = load_data()
    
    # 2. Проверка аномалий
    train_anomalies = check_anomalies(train_df, "Train")
    test_anomalies = check_anomalies(test_df, "Test")
    
    # 3. Исправление аномалий
    train_df_clean = fix_anomalies(train_df)
    test_df_clean = fix_anomalies(test_df)
    
    # 4. Проверка пропусков
    check_missing_values(train_df_clean)
    
    # 5. Проверка дубликатов
    check_duplicates(train_df_clean)
    
    # 6. Анализ выбросов
    outliers = analyze_outliers(train_df_clean)
    
    # 7. Обработка скошенности
    train_df_clean = handle_skewness(train_df_clean)
    test_df_clean = handle_skewness(test_df_clean)
    
    # 8. Сохранение очищенных данных
    train_df_clean.to_csv(DATA_PROCESSED / 'train_cleaned.csv', index=False)
    test_df_clean.to_csv(DATA_PROCESSED / 'test_cleaned.csv', index=False)
    print(f"\n💾 Очищенные данные сохранены в {DATA_PROCESSED}")
    
    # 9. Запуск baseline v1
    roc_auc, improvement = run_baseline_v1(train_df_clean)
    
    # 10. Итоговый отчет
    print("\n" + "="*60)
    print("STEP 3 COMPLETED!")
    print("="*60)
    print(f"✅ Аномалии исправлены")
    print(f"✅ Скошенность обработана")
    print(f"✅ Baseline v1: ROC-AUC = {roc_auc:.4f} ({improvement:+.4f})")
    print(f"✅ Результаты сохранены в {MODEL_V1}")

if __name__ == "__main__":
    main()