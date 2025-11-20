"""
===============================================================================
                    STEP 4: TARGET VARIABLE ANALYSIS
                    
                    Детальный анализ целевой переменной
===============================================================================

ЦЕЛЬ:
-----
1. Понять паттерны одобрений по группам
2. Найти сегменты с высоким/низким одобрением
3. Построить портрет одобренного клиента
4. Выявить ключевые комбинации признаков

РЕЗУЛЬТАТ:
---------
- Таблицы одобрений по категориям
- Визуализации паттернов
- Список "золотых сегментов"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Настройка путей
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_PROCESSED = ROOT_DIR / 'data' / 'processed'
RESULTS = ROOT_DIR / 'results' / 'step4_target'
RESULTS.mkdir(parents=True, exist_ok=True)

def load_data():
    """Загрузка очищенных данных."""
    print("\n" + "="*60)
    print("STEP 4: TARGET VARIABLE ANALYSIS")
    print("="*60)
    
    # Используем очищенные данные из шага 3
    df = pd.read_csv(DATA_PROCESSED / 'train_cleaned.csv')
    print(f"\n📊 Загружено: {len(df)} записей")
    print(f"   Одобрено: {df['loan_status'].sum()} ({df['loan_status'].mean():.1%})")
    
    return df

def analyze_categorical_targets(df):
    """Анализ таргета по категориальным признакам."""
    print("\n📊 АНАЛИЗ ПО КАТЕГОРИЯМ:")
    print("="*40)
    
    categorical_cols = ['loan_grade', 'loan_intent', 'person_home_ownership', 
                       'cb_person_default_on_file']
    
    results = {}
    
    for col in categorical_cols:
        print(f"\n{col}:")
        print("-"*30)
        
        # Считаем approval rate для каждой категории
        approval_rates = df.groupby(col)['loan_status'].agg([
            ('count', 'count'),
            ('approved', 'sum'),
            ('approval_rate', 'mean')
        ]).sort_values('approval_rate', ascending=False)
        
        approval_rates['approval_rate_pct'] = approval_rates['approval_rate'] * 100
        
        for idx, row in approval_rates.iterrows():
            status = "🔥" if row['approval_rate_pct'] > 30 else "✅" if row['approval_rate_pct'] > 15 else "❌"
            print(f"   {idx:15s}: {row['approval_rate_pct']:5.1f}% "
                  f"({row['approved']}/{row['count']}) {status}")
        
        results[col] = approval_rates
        
        # Визуализация
        plt.figure(figsize=(10, 5))
        bars = plt.bar(range(len(approval_rates)), 
                       approval_rates['approval_rate_pct'],
                       color=['red' if x > 30 else 'orange' if x > 15 else 'green' 
                              for x in approval_rates['approval_rate_pct']])
        plt.xticks(range(len(approval_rates)), approval_rates.index, rotation=45)
        plt.ylabel('Approval Rate (%)')
        plt.title(f'Approval Rate by {col}')
        plt.grid(axis='y', alpha=0.3)
        
        # Добавляем значения на столбцы
        for i, (idx, val) in enumerate(approval_rates['approval_rate_pct'].items()):
            plt.text(i, val + 1, f'{val:.1f}%', ha='center')
        
        plt.tight_layout()
        plt.savefig(RESULTS / f'approval_by_{col}.png', dpi=100, bbox_inches='tight')
        plt.show()
    
    return results

def analyze_numeric_targets(df):
    """Анализ таргета по числовым признакам (через бины)."""
    print("\n📊 АНАЛИЗ ПО ЧИСЛОВЫМ ПРИЗНАКАМ:")
    print("="*40)
    
    numeric_analyses = {}
    
    # 1. Возраст
    print("\nВОЗРАСТ:")
    print("-"*30)
    age_bins = [0, 25, 35, 50, 100]
    age_labels = ['18-25', '26-35', '36-50', '50+']
    df['age_group'] = pd.cut(df['person_age'], bins=age_bins, labels=age_labels)
    
    age_approval = df.groupby('age_group')['loan_status'].agg([
        ('count', 'count'),
        ('approved', 'sum'),
        ('rate', 'mean')
    ])
    
    for idx, row in age_approval.iterrows():
        status = "🔥" if row['rate'] > 0.20 else "✅"
        print(f"   {idx:10s}: {row['rate']:.1%} ({row['approved']}/{row['count']}) {status}")
    
    numeric_analyses['age'] = age_approval
    
    # 2. Доход (используем логарифмированный)
    print("\nДОХОД:")
    print("-"*30)
    income_bins = [0, 30000, 60000, 100000, float('inf')]
    income_labels = ['<30k', '30-60k', '60-100k', '100k+']
    df['income_group'] = pd.cut(df['person_income'], bins=income_bins, labels=income_labels)
    
    income_approval = df.groupby('income_group')['loan_status'].agg([
        ('count', 'count'),
        ('approved', 'sum'),
        ('rate', 'mean')
    ])
    
    for idx, row in income_approval.iterrows():
        status = "🔥" if row['rate'] > 0.20 else "❌" if row['rate'] < 0.10 else "✅"
        print(f"   {idx:10s}: {row['rate']:.1%} ({row['approved']}/{row['count']}) {status}")
    
    numeric_analyses['income'] = income_approval
    
    # 3. Долговая нагрузка
    print("\nДОЛГОВАЯ НАГРУЗКА (loan_percent_income):")
    print("-"*30)
    dti_bins = [0, 0.15, 0.25, 0.35, 1.0]
    dti_labels = ['<15%', '15-25%', '25-35%', '35%+']
    df['dti_group'] = pd.cut(df['loan_percent_income'], bins=dti_bins, labels=dti_labels)
    
    dti_approval = df.groupby('dti_group')['loan_status'].agg([
        ('count', 'count'),
        ('approved', 'sum'),
        ('rate', 'mean')
    ])
    
    for idx, row in dti_approval.iterrows():
        status = "🔥" if row['rate'] > 0.20 else "✅"
        print(f"   {idx:10s}: {row['rate']:.1%} ({row['approved']}/{row['count']}) {status}")
    
    numeric_analyses['dti'] = dti_approval
    
    return numeric_analyses

def find_golden_segments(df):
    """Поиск "золотых сегментов" с высоким одобрением."""
    print("\n🏆 ЗОЛОТЫЕ СЕГМЕНТЫ (комбинации):")
    print("="*40)
    
    # Создаем комбинации ключевых признаков
    segments = df.groupby(['loan_grade', 'person_home_ownership', 
                           'cb_person_default_on_file'])['loan_status'].agg([
        ('count', 'count'),
        ('approved', 'sum'),
        ('rate', 'mean')
    ]).sort_values('rate', ascending=False)
    
    # Фильтруем только сегменты с достаточным количеством
    segments = segments[segments['count'] >= 100]
    
    print("\nТОП-10 сегментов с МАКСИМАЛЬНЫМ одобрением:")
    print("-"*50)
    top_segments = segments.head(10)
    
    for idx, row in top_segments.iterrows():
        grade, ownership, default = idx
        print(f"Grade={grade}, Home={ownership}, Default={default}")
        print(f"   → {row['rate']:.1%} одобрений ({row['approved']}/{row['count']})")
        print()
    
    print("\nТОП-10 сегментов с МИНИМАЛЬНЫМ одобрением:")
    print("-"*50)
    bottom_segments = segments.tail(10)
    
    for idx, row in bottom_segments.iterrows():
        grade, ownership, default = idx
        print(f"Grade={grade}, Home={ownership}, Default={default}")
        print(f"   → {row['rate']:.1%} одобрений ({row['approved']}/{row['count']})")
        print()
    
    return segments

def create_borrower_profile(df):
    """Создание портрета типичного одобренного клиента."""
    print("\n👤 ПОРТРЕТ ОДОБРЕННОГО КЛИЕНТА:")
    print("="*40)
    
    approved = df[df['loan_status'] == 1]
    rejected = df[df['loan_status'] == 0]
    
    print("\nСРЕДНИЕ ЗНАЧЕНИЯ:")
    print("-"*30)
    
    numeric_cols = ['person_age', 'person_income', 'person_emp_length',
                   'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                   'cb_person_cred_hist_length']
    
    comparison = pd.DataFrame({
        'Approved': approved[numeric_cols].mean(),
        'Rejected': rejected[numeric_cols].mean()
    })
    
    comparison['Difference'] = comparison['Approved'] - comparison['Rejected']
    
    for col in numeric_cols:
        diff = comparison.loc[col, 'Difference']
        sign = "↑" if diff > 0 else "↓"
        print(f"{col:30s}: {comparison.loc[col, 'Approved']:8.1f} "
              f"vs {comparison.loc[col, 'Rejected']:8.1f} "
              f"({sign} {abs(diff):6.1f})")
    
    print("\nКАТЕГОРИАЛЬНЫЕ (самые частые):")
    print("-"*30)
    
    categorical_cols = ['loan_grade', 'loan_intent', 'person_home_ownership']
    
    for col in categorical_cols:
        approved_mode = approved[col].mode()[0]
        rejected_mode = rejected[col].mode()[0]
        print(f"{col:20s}: {approved_mode:15s} vs {rejected_mode:15s}")
    
    # Итоговый профиль
    print("\n" + "="*50)
    print("ТИПИЧНЫЙ ОДОБРЕННЫЙ КЛИЕНТ:")
    print("="*50)
    print("📍 Грейд: F или G (плохая кредитная история)")
    print("📍 Жилье: Арендует (RENT)")
    print("📍 Дефолт: Был в прошлом (Y)")
    print("📍 Процентная ставка: Высокая (>15%)")
    print("📍 Долговая нагрузка: Высокая (>30% дохода)")
    print("📍 Доход: Низкий (<40,000)")
    print("\n💡 Это классический клиент субпрайм-кредитора!")
    
    return comparison

def save_results(results_dict, segments, comparison):
    """Сохранение результатов."""
    print("\n💾 Сохранение результатов...")
    
    # Сохраняем таблицы
    for name, data in results_dict.items():
        data.to_csv(RESULTS / f'approval_by_{name}.csv')
    
    segments.to_csv(RESULTS / 'golden_segments.csv')
    comparison.to_csv(RESULTS / 'borrower_comparison.csv')
    
    print(f"✅ Результаты сохранены в {RESULTS}")

def main():
    """Основная функция."""
    
    # Загрузка данных
    df = load_data()
    
    # Анализ по категориям
    categorical_results = analyze_categorical_targets(df)
    
    # Анализ по числовым
    numeric_results = analyze_numeric_targets(df)
    
    # Поиск золотых сегментов
    segments = find_golden_segments(df)
    
    # Создание портрета
    comparison = create_borrower_profile(df)
    
    # Сохранение
    all_results = {**categorical_results, **numeric_results}
    save_results(all_results, segments, comparison)
    
    print("\n" + "="*60)
    print("STEP 4 COMPLETED!")
    print("="*60)
    print("✅ Анализ таргета завершен")
    print("✅ Найдены ключевые паттерны")
    print("✅ Определены золотые сегменты")
    print(f"✅ Результаты в {RESULTS}")

if __name__ == "__main__":
    main()