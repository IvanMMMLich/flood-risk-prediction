"""
===============================================================================
                    💳 LOAN APPROVAL PREDICTION
                    
                 ШАГ 1: ПЕРВОЕ ЗНАКОМСТВО С ДАННЫМИ
                        "Understanding Credit Data"
===============================================================================

КОНТЕКСТ ЗАДАЧИ:
----------------
Банк хочет автоматизировать процесс одобрения кредитов. Нужно построить
модель, которая на основе информации о клиенте предскажет, будет ли
кредит одобрен или отклонен.

ЧТО МЫ УЗНАЕМ В ЭТОМ ФАЙЛЕ:
--------------------------
1. Размер и структура данных
2. Типы признаков (числовые vs категориальные)
3. Первый взгляд на целевую переменную
4. Базовая статистика
"""

import pandas as pd
import numpy as np
from pathlib import Path
from colorama import init, Fore, Style
import warnings

warnings.filterwarnings('ignore')
init(autoreset=True)

# Пути
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_RAW = ROOT_DIR / 'data' / 'raw'
RESULTS = ROOT_DIR / 'results'

def load_data():
    """Загрузка данных о кредитах."""
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{Fore.CYAN} ЗАГРУЗКА ДАННЫХ О КРЕДИТАХ")
    print(f"{Fore.CYAN}{'='*80}")
    
    train_path = DATA_RAW / 'train.csv'
    test_path = DATA_RAW / 'test.csv'
    
    print(f"\n⏳ Загружаю данные...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"{Fore.GREEN} Train загружен: {train_df.shape}")
    print(f"{Fore.GREEN} Test загружен: {test_df.shape}")
    
    return train_df, test_df

def analyze_structure(train_df, test_df):
    """Анализ структуры данных о кредитах."""
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{Fore.CYAN} СТРУКТУРА ДАННЫХ О КРЕДИТАХ")
    print(f"{Fore.CYAN}{'='*80}")
    
    # Классификация столбцов
    personal_features = ['person_age', 'person_income', 'person_home_ownership', 'person_emp_length']
    loan_features = ['loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate', 'loan_percent_income']
    credit_history = ['cb_person_default_on_file', 'cb_person_cred_hist_length']
    
    print(f"\n👤 Персональные данные клиента:")
    for col in personal_features:
        if col in train_df.columns:
            dtype = train_df[col].dtype
            unique = train_df[col].nunique()
            print(f"   • {col:30} | Тип: {dtype} | Уникальных: {unique}")
    
    print(f"\n💰 Параметры кредита:")
    for col in loan_features:
        if col in train_df.columns:
            dtype = train_df[col].dtype
            unique = train_df[col].nunique()
            print(f"   • {col:30} | Тип: {dtype} | Уникальных: {unique}")
    
    print(f"\n Кредитная история:")
    for col in credit_history:
        if col in train_df.columns:
            dtype = train_df[col].dtype
            unique = train_df[col].nunique()
            print(f"   • {col:30} | Тип: {dtype} | Уникальных: {unique}")
    
    print(f"\n Целевая переменная:")
    print(f"   • loan_status (0=отказ, 1=одобрение)")

def analyze_target(train_df):
    """Анализ целевой переменной loan_status."""
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{Fore.CYAN}🎯 АНАЛИЗ ОДОБРЕНИЯ КРЕДИТОВ")
    print(f"{Fore.CYAN}{'='*80}")
    
    target_counts = train_df['loan_status'].value_counts()
    target_pct = train_df['loan_status'].value_counts(normalize=True) * 100
    
    print(f"\n📊 Распределение loan_status:")
    print(f"   Отказано (0): {target_counts[0]:,} ({target_pct[0]:.1f}%)")
    print(f"   Одобрено (1): {target_counts[1]:,} ({target_pct[1]:.1f}%)")
    
    approval_rate = target_pct[1]
    print(f"\n💡 Общий процент одобрения: {approval_rate:.1f}%")
    
    if approval_rate < 30:
        print(f"   {Fore.RED} Очень низкий процент одобрения - сильный дисбаланс!")
    elif approval_rate < 40:
        print(f"   {Fore.YELLOW} Низкий процент одобрения - есть дисбаланс")
    else:
        print(f"   {Fore.GREEN} Умеренный процент одобрения")

def analyze_features(train_df):
    """Первичный анализ признаков."""
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{Fore.CYAN} БЫСТРЫЙ АНАЛИЗ ПРИЗНАКОВ")
    print(f"{Fore.CYAN}{'='*80}")
    
    # Числовые признаки
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('id')
    if 'loan_status' in numeric_cols:
        numeric_cols.remove('loan_status')
    
    print(f"\n Числовые признаки:")
    for col in numeric_cols:
        mean_val = train_df[col].mean()
        median_val = train_df[col].median()
        min_val = train_df[col].min()
        max_val = train_df[col].max()
        null_count = train_df[col].isnull().sum()
        
        print(f"\n   {col}:")
        print(f"      Диапазон: [{min_val:.1f} - {max_val:.1f}]")
        print(f"      Среднее: {mean_val:.1f}, Медиана: {median_val:.1f}")
        if null_count > 0:
            print(f"      {Fore.YELLOW}Пропуски: {null_count} ({null_count/len(train_df)*100:.1f}%)")
    
    # Категориальные признаки
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        print(f"\n Категориальные признаки:")
        for col in categorical_cols:
            unique_values = train_df[col].nunique()
            top_value = train_df[col].value_counts().index[0]
            top_count = train_df[col].value_counts().values[0]
            null_count = train_df[col].isnull().sum()
            
            print(f"\n   {col}:")
            print(f"      Уникальных значений: {unique_values}")
            print(f"      Самое частое: {top_value} ({top_count/len(train_df)*100:.1f}%)")
            if null_count > 0:
                print(f"      {Fore.YELLOW}Пропуски: {null_count} ({null_count/len(train_df)*100:.1f}%)")

def main():
    """Главная функция."""
    print(f"{Fore.MAGENTA}{'='*80}")
    print(f"{Fore.MAGENTA}{' '*25}💳 LOAN APPROVAL PREDICTION")
    print(f"{Fore.MAGENTA}{' '*25}Шаг 1: Первое знакомство")
    print(f"{Fore.MAGENTA}{'='*80}")
    
    # Загрузка
    train_df, test_df = load_data()
    
    # Анализ структуры
    analyze_structure(train_df, test_df)
    
    # Анализ целевой переменной
    analyze_target(train_df)
    
    # Анализ признаков
    analyze_features(train_df)
    
    print(f"\n{Fore.MAGENTA}{'='*80}")
    print(f"{Fore.MAGENTA}{' '*30} ШАГ 1 ЗАВЕРШЕН!")
    print(f"{Fore.MAGENTA}{'='*80}")
    
    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = main()

    