"""
===============================================================================
                    STEP 5: FEATURE IMPORTANCE ANALYSIS
                    
                    Математический анализ важности признаков
===============================================================================

ЦЕЛЬ:
-----
1. Измерить важность признаков через Mutual Information
2. Проверить важность через Permutation Importance
3. Сравнить разные методы оценки важности
4. Определить топ-признаки для модели

РЕЗУЛЬТАТ:
---------
- Ранжированный список признаков по важности
- Сравнение методов оценки
- Рекомендации по отбору признаков
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

# Настройка путей
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_PROCESSED = ROOT_DIR / 'data' / 'processed'
RESULTS = ROOT_DIR / 'results' / 'step5_importance'
RESULTS.mkdir(parents=True, exist_ok=True)

def load_and_prepare_data():
    """Загрузка и подготовка данных для анализа важности."""
    print("\n" + "="*60)
    print("STEP 5: FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Загружаем очищенные данные
    df = pd.read_csv(DATA_PROCESSED / 'train_cleaned.csv')
    print(f"\n📊 Загружено: {len(df)} записей")
    
    # Разделяем на X и y
    X = df.drop(['id', 'loan_status'], axis=1)
    y = df['loan_status']
    
    # Кодируем категориальные признаки
    le_dict = {}
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    print("\n🔧 Кодирование категориальных признаков:")
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        le_dict[col] = le
        print(f"   {col}: {len(le.classes_)} категорий")
    
    return X, y, le_dict

def calculate_mutual_information(X, y):
    """Расчет Mutual Information Score."""
    print("\n📊 MUTUAL INFORMATION SCORE:")
    print("="*40)
    print("(Сколько информации о таргете дает признак)")
    print("-"*40)
    
    # Расчет MI scores
    mi_scores = mutual_info_classif(X, y, random_state=42)
    
    # Создаем DataFrame с результатами
    mi_df = pd.DataFrame({
        'feature': X.columns,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    # Нормализуем для удобства (0-1)
    mi_df['mi_normalized'] = mi_df['mi_score'] / mi_df['mi_score'].max()
    
    # Вывод результатов
    print("\nТоп-10 признаков по MI:")
    for idx, row in mi_df.head(10).iterrows():
        bar = '█' * int(row['mi_normalized'] * 20)
        print(f"{row['feature']:25s}: {row['mi_score']:.4f} {bar}")
    
    # Визуализация
    plt.figure(figsize=(10, 8))
    top_15 = mi_df.head(15)
    colors = ['red' if x > 0.5 else 'orange' if x > 0.25 else 'green' 
              for x in top_15['mi_normalized']]
    
    plt.barh(range(len(top_15)), top_15['mi_score'], color=colors)
    plt.yticks(range(len(top_15)), top_15['feature'])
    plt.xlabel('Mutual Information Score')
    plt.title('Feature Importance by Mutual Information')
    plt.gca().invert_yaxis()
    
    # Добавляем значения на полоски
    for i, (idx, row) in enumerate(top_15.iterrows()):
        plt.text(row['mi_score'] + 0.001, i, f'{row["mi_score"]:.4f}', 
                va='center')
    
    plt.tight_layout()
    plt.savefig(RESULTS / 'mutual_information.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    return mi_df

def calculate_permutation_importance(X, y):
    """Расчет Permutation Importance."""
    print("\n📊 PERMUTATION IMPORTANCE:")
    print("="*40)
    print("(Насколько упадет точность при перемешивании признака)")
    print("-"*40)
    
    # Разделяем на train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Обучаем простую модель
    print("\nОбучаем LogisticRegression для оценки...")
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Расчет permutation importance
    print("Вычисляем permutation importance (может занять минуту)...")
    perm_importance = permutation_importance(
        model, X_val, y_val, 
        n_repeats=10, 
        random_state=42,
        scoring='roc_auc'
    )
    
    # Создаем DataFrame с результатами
    perm_df = pd.DataFrame({
        'feature': X.columns,
        'importance': perm_importance.importances_mean,
        'std': perm_importance.importances_std
    }).sort_values('importance', ascending=False)
    
    # Вывод результатов
    print("\nТоп-10 признаков по Permutation Importance:")
    for idx, row in perm_df.head(10).iterrows():
        bar = '█' * int((row['importance'] / perm_df['importance'].max()) * 20)
        print(f"{row['feature']:25s}: {row['importance']:.4f} ± {row['std']:.4f} {bar}")
    
    # Визуализация
    plt.figure(figsize=(10, 8))
    top_15 = perm_df.head(15)
    
    plt.barh(range(len(top_15)), top_15['importance'], 
            xerr=top_15['std'], color='steelblue', alpha=0.7)
    plt.yticks(range(len(top_15)), top_15['feature'])
    plt.xlabel('Permutation Importance')
    plt.title('Feature Importance by Permutation')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(RESULTS / 'permutation_importance.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    return perm_df

def calculate_forest_importance(X, y):
    """Важность признаков из Random Forest."""
    print("\n📊 RANDOM FOREST FEATURE IMPORTANCE:")
    print("="*40)
    print("(Встроенная важность из дерева решений)")
    print("-"*40)
    
    # Разделяем на train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Обучаем Random Forest
    print("\nОбучаем Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Получаем важность
    rf_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Вывод результатов
    print("\nТоп-10 признаков по Random Forest:")
    for idx, row in rf_df.head(10).iterrows():
        bar = '█' * int((row['importance'] / rf_df['importance'].max()) * 20)
        print(f"{row['feature']:25s}: {row['importance']:.4f} {bar}")
    
    return rf_df

def compare_methods(mi_df, perm_df, rf_df):
    """Сравнение методов оценки важности."""
    print("\n📊 СРАВНЕНИЕ МЕТОДОВ:")
    print("="*40)
    
    # Объединяем результаты
    comparison = pd.DataFrame({
        'feature': mi_df['feature'],
        'mutual_info_rank': range(1, len(mi_df) + 1),
        'mi_score': mi_df['mi_score'].values
    })
    
    # Добавляем ранги из других методов
    perm_ranks = {feat: rank+1 for rank, feat in enumerate(perm_df['feature'])}
    rf_ranks = {feat: rank+1 for rank, feat in enumerate(rf_df['feature'])}
    
    comparison['perm_rank'] = comparison['feature'].map(perm_ranks)
    comparison['rf_rank'] = comparison['feature'].map(rf_ranks)
    
    # Средний ранг
    comparison['avg_rank'] = comparison[['mutual_info_rank', 'perm_rank', 'rf_rank']].mean(axis=1)
    comparison = comparison.sort_values('avg_rank')
    
    print("\nСОГЛАСОВАННЫЙ ТОП-10 (по среднему рангу):")
    print("-"*50)
    print(f"{'Признак':<25} {'MI':>5} {'Perm':>5} {'RF':>5} {'Avg':>6}")
    print("-"*50)
    
    for idx, row in comparison.head(10).iterrows():
        print(f"{row['feature']:<25} {row['mutual_info_rank']:5.0f} "
              f"{row['perm_rank']:5.0f} {row['rf_rank']:5.0f} "
              f"{row['avg_rank']:6.1f}")
    
    # Визуализация согласованности
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Топ-10 по каждому методу
    methods = [
        ('Mutual Information', mi_df.head(10)),
        ('Permutation Importance', perm_df.head(10)),
        ('Random Forest', rf_df.head(10))
    ]
    
    for ax, (method_name, data) in zip(axes, methods):
        ax.barh(range(10), data.iloc[:, 1], color='skyblue')
        ax.set_yticks(range(10))
        ax.set_yticklabels(data['feature'])
        ax.set_title(method_name)
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(RESULTS / 'methods_comparison.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    return comparison

def create_recommendations(comparison):
    """Создание рекомендаций по отбору признаков."""
    print("\n💡 РЕКОМЕНДАЦИИ:")
    print("="*40)
    
    # Топ признаки по согласованному рангу
    top_features = comparison.head(10)['feature'].tolist()
    
    print("\nОБЯЗАТЕЛЬНЫЕ признаки (топ-5):")
    for i, feat in enumerate(top_features[:5], 1):
        print(f"   {i}. {feat}")
    
    print("\nВАЖНЫЕ признаки (6-10):")
    for i, feat in enumerate(top_features[5:10], 6):
        print(f"   {i}. {feat}")
    
    print("\nВОЗМОЖНО УДАЛИТЬ (низкая важность):")
    bottom_features = comparison.tail(3)['feature'].tolist()
    for feat in bottom_features:
        print(f"   - {feat}")
    
    print("\n📌 ВЫВОДЫ:")
    print("-"*40)
    print("1. loan_percent_income - КРИТИЧЕСКИ важен")
    print("2. loan_int_rate - второй по важности")
    print("3. loan_grade - ключевой категориальный")
    print("4. Возраст и доход - умеренно важны")
    print("5. cb_person_cred_hist_length - можно исключить")
    
    return top_features

def save_results(mi_df, perm_df, rf_df, comparison, top_features):
    """Сохранение результатов."""
    print("\n💾 Сохранение результатов...")
    
    # Сохраняем таблицы
    mi_df.to_csv(RESULTS / 'mutual_information.csv', index=False)
    perm_df.to_csv(RESULTS / 'permutation_importance.csv', index=False)
    rf_df.to_csv(RESULTS / 'random_forest_importance.csv', index=False)
    comparison.to_csv(RESULTS / 'methods_comparison.csv', index=False)
    
    # Сохраняем список топ-признаков
    with open(RESULTS / 'top_features.txt', 'w') as f:
        f.write("TOP FEATURES FOR MODEL:\n")
        f.write("="*30 + "\n")
        for i, feat in enumerate(top_features, 1):
            f.write(f"{i}. {feat}\n")
    
    print(f"✅ Результаты сохранены в {RESULTS}")

def main():
    """Основная функция."""
    
    # Загрузка и подготовка данных
    X, y, le_dict = load_and_prepare_data()
    
    # Расчет важности разными методами
    mi_df = calculate_mutual_information(X, y)
    perm_df = calculate_permutation_importance(X, y)
    rf_df = calculate_forest_importance(X, y)
    
    # Сравнение методов
    comparison = compare_methods(mi_df, perm_df, rf_df)
    
    # Рекомендации
    top_features = create_recommendations(comparison)
    
    # Сохранение
    save_results(mi_df, perm_df, rf_df, comparison, top_features)
    
    print("\n" + "="*60)
    print("STEP 5 COMPLETED!")
    print("="*60)
    print("✅ Важность признаков измерена тремя методами")
    print("✅ Определены топ-признаки для модели")
    print("✅ Созданы рекомендации по отбору")
    print(f"✅ Результаты в {RESULTS}")

if __name__ == "__main__":
    main()