"""
===============================================================================
                    💳 LOAN APPROVAL PREDICTION
                    
                 ШАГ 2: ДЕТАЛЬНОЕ ИЗУЧЕНИЕ ПРИЗНАКОВ
                    "Deep Dive into Each Feature"
===============================================================================

ЦЕЛЬ ЭТОГО ФАЙЛА:
-----------------
Детально изучить КАЖДЫЙ признак по отдельности. В прошлый раз мы увидели
общую картину, теперь копаем глубже - смотрим на распределения, выбросы,
связь с целевой переменной.

ЧТО МЫ УЗНАЕМ:
-------------
1. Распределение каждого признака (визуализация)
2. Выбросы и аномалии (те самые 123 года!)
3. Как каждый признак влияет на одобрение кредита
4. Какие признаки самые важные

ВАЖНО:
------
Создадим отдельную папку для КАЖДОГО признака с детальным анализом!
"""

# ==============================================================================
# ИМПОРТЫ
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from colorama import init, Fore, Style
import warnings

# Настройки для красивого вывода
warnings.filterwarnings('ignore')
init(autoreset=True)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==============================================================================
# НАСТРОЙКА ПУТЕЙ
# ==============================================================================

# Корневая директория проекта
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_RAW = ROOT_DIR / 'data' / 'raw'
RESULTS = ROOT_DIR / 'results'

# Создаем папку для результатов этого шага
STEP2_DIR = RESULTS / 'step2_deep_explore'
STEP2_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# ФУНКЦИЯ 1: ЗАГРУЗКА ДАННЫХ
# ==============================================================================

def load_data():
    """
    Загружает данные о кредитах.
    
    Почему отдельная функция:
    - Переиспользование в разных файлах
    - Единообразная загрузка везде
    """
    print(f"\n{Fore.CYAN}📂 Загружаю данные...")
    
    train_df = pd.read_csv(DATA_RAW / 'train.csv')
    test_df = pd.read_csv(DATA_RAW / 'test.csv')
    
    print(f"{Fore.GREEN}✅ Загружено: {len(train_df):,} train, {len(test_df):,} test")
    return train_df, test_df

# ==============================================================================
# ФУНКЦИЯ 2: КЛАССИФИКАЦИЯ ПРИЗНАКОВ
# ==============================================================================

def classify_features(train_df):
    """
    Правильно разделяет признаки на группы для анализа.
    
    Группировка помогает:
    - Понять бизнес-логику
    - Применять разные методы обработки
    - Создавать осмысленные feature interactions
    """
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{Fore.CYAN}📊 КЛАССИФИКАЦИЯ ПРИЗНАКОВ")
    print(f"{Fore.CYAN}{'='*80}")
    
    # Определяем группы признаков по смыслу
    personal_features = [
        'person_age',           # Возраст клиента
        'person_income',        # Годовой доход
        'person_home_ownership',# Тип владения жильем
        'person_emp_length'     # Стаж работы
    ]
    
    loan_features = [
        'loan_intent',          # Цель кредита
        'loan_grade',           # Грейд кредита (риск)
        'loan_amnt',            # Сумма кредита
        'loan_int_rate',        # Процентная ставка
        'loan_percent_income'   # Платеж/доход
    ]
    
    credit_history = [
        'cb_person_default_on_file',      # Был ли дефолт
        'cb_person_cred_hist_length'       # Длина кредитной истории
    ]
    
    # Определяем типы данных для каждого признака
    numeric_features = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [f for f in numeric_features if f not in ['id', 'loan_status']]
    
    categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\n👤 Персональные данные: {len(personal_features)} признаков")
    print(f"💰 Параметры кредита: {len(loan_features)} признаков")
    print(f"📊 Кредитная история: {len(credit_history)} признаков")
    print(f"\n🔢 Числовые признаки: {len(numeric_features)}")
    print(f"📝 Категориальные признаки: {len(categorical_features)}")
    
    return {
        'personal': personal_features,
        'loan': loan_features,
        'credit': credit_history,
        'numeric': numeric_features,
        'categorical': categorical_features
    }

# ==============================================================================
# ФУНКЦИЯ 3: АНАЛИЗ ЧИСЛОВЫХ ПРИЗНАКОВ
# ==============================================================================

def analyze_numeric_features(train_df, numeric_features):
    """
    Детальный анализ каждого числового признака.
    
    Что анализируем:
    - Распределение (нормальное или скошенное?)
    - Выбросы (IQR метод)
    - Связь с целевой переменной
    """
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{Fore.CYAN} АНАЛИЗ ЧИСЛОВЫХ ПРИЗНАКОВ")
    print(f"{Fore.CYAN}{'='*80}")
    
    # Создаем папку для числовых признаков
    numeric_dir = STEP2_DIR / 'numeric_features'
    numeric_dir.mkdir(parents=True, exist_ok=True)
    
    # Список для сохранения статистик
    numeric_stats = []
    
    for feature in numeric_features:
        print(f"\n Анализирую {feature}...")
        
        # Создаем папку для этого признака
        feature_dir = numeric_dir / feature
        feature_dir.mkdir(parents=True, exist_ok=True)
        
        # Получаем данные
        data = train_df[feature].dropna()  # Убираем пропуски для анализа
        
        # ========== СЧИТАЕМ СТАТИСТИКИ ==========
        stats = {
            'feature': feature,
            'count': len(data),
            'missing': train_df[feature].isnull().sum(),
            'missing_pct': train_df[feature].isnull().sum() / len(train_df) * 100,
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'q25': data.quantile(0.25),
            'q75': data.quantile(0.75),
            'iqr': data.quantile(0.75) - data.quantile(0.25),
            'skewness': data.skew(),
            'kurtosis': data.kurtosis()
        }
        
        # ========== НАХОДИМ ВЫБРОСЫ (IQR метод) ==========
        Q1 = stats['q25']
        Q3 = stats['q75']
        IQR = stats['iqr']
        
        # Границы для выбросов
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Считаем выбросы
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        stats['outliers_count'] = len(outliers)
        stats['outliers_pct'] = len(outliers) / len(data) * 100
        
        # Особые проверки для возраста и стажа (те самые 123!)
        if feature in ['person_age', 'person_emp_length']:
            suspicious_123 = (train_df[feature] == 123).sum()
            if suspicious_123 > 0:
                stats['suspicious_123'] = suspicious_123
                print(f"   {Fore.YELLOW} Найдено {suspicious_123} значений = 123 (возможно, код для 'неизвестно')")
        
        numeric_stats.append(stats)
        
        # ========== ВИЗУАЛИЗАЦИЯ ==========
        # Создаем фигуру с 4 графиками для каждого признака
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # График 1: Гистограмма с линиями среднего и медианы
        axes[0, 0].hist(data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 0].axvline(data.mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {data.mean():.1f}')
        axes[0, 0].axvline(data.median(), color='green', linestyle='--', 
                          linewidth=2, label=f'Median: {data.median():.1f}')
        axes[0, 0].set_title(f'{feature} - Distribution')
        axes[0, 0].set_xlabel(feature)
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # График 2: Boxplot для выбросов
        bp = axes[0, 1].boxplot(data, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][0].set_alpha(0.7)
        
        # Добавляем информацию о выбросах
        axes[0, 1].set_title(f'{feature} - Boxplot\nOutliers: {stats["outliers_count"]} ({stats["outliers_pct"]:.1f}%)')
        axes[0, 1].set_ylabel(feature)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Добавляем линии для квартилей
        axes[0, 1].text(1.1, stats['min'], f'Min: {stats["min"]:.1f}', fontsize=9)
        axes[0, 1].text(1.1, stats['q25'], f'Q1: {stats["q25"]:.1f}', fontsize=9)
        axes[0, 1].text(1.1, stats['median'], f'Median: {stats["median"]:.1f}', fontsize=9)
        axes[0, 1].text(1.1, stats['q75'], f'Q3: {stats["q75"]:.1f}', fontsize=9)
        axes[0, 1].text(1.1, stats['max'], f'Max: {stats["max"]:.1f}', fontsize=9)
        
        # График 3: Распределение по loan_status
        # Сравниваем распределение для одобренных и отклоненных
        approved = train_df[train_df['loan_status'] == 1][feature].dropna()
        rejected = train_df[train_df['loan_status'] == 0][feature].dropna()
        
        axes[1, 0].hist([rejected, approved], bins=20, alpha=0.7, 
                       label=['Rejected', 'Approved'], color=['red', 'green'])
        axes[1, 0].set_title(f'{feature} by Loan Status')
        axes[1, 0].set_xlabel(feature)
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # График 4: Violin plot для сравнения распределений
        df_plot = train_df[[feature, 'loan_status']].dropna()
        df_plot['Status'] = df_plot['loan_status'].map({0: 'Rejected', 1: 'Approved'})
        
        sns.violinplot(data=df_plot, x='Status', y=feature, ax=axes[1, 1])
        axes[1, 1].set_title(f'{feature} Distribution by Status')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Общий заголовок
        fig.suptitle(f'Детальный анализ: {feature}', fontsize=14, y=1.02)
        plt.tight_layout()
        
        # Сохраняем график
        plt.savefig(feature_dir / f'{feature}_analysis.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # ========== СОХРАНЯЕМ СТАТИСТИКИ В ФАЙЛ ==========
        with open(feature_dir / f'{feature}_stats.txt', 'w') as f:
            f.write(f"Статистика для признака: {feature}\n")
            f.write("="*50 + "\n\n")
            
            f.write("ОСНОВНЫЕ МЕТРИКИ:\n")
            f.write(f"  Количество: {stats['count']:,}\n")
            f.write(f"  Пропуски: {stats['missing']:,} ({stats['missing_pct']:.1f}%)\n")
            f.write(f"  Среднее: {stats['mean']:.2f}\n")
            f.write(f"  Медиана: {stats['median']:.2f}\n")
            f.write(f"  Std: {stats['std']:.2f}\n")
            f.write(f"  Min: {stats['min']:.2f}\n")
            f.write(f"  Max: {stats['max']:.2f}\n")
            
            f.write("\nКВАРТИЛИ:\n")
            f.write(f"  Q1 (25%): {stats['q25']:.2f}\n")
            f.write(f"  Q2 (50%): {stats['median']:.2f}\n")
            f.write(f"  Q3 (75%): {stats['q75']:.2f}\n")
            f.write(f"  IQR: {stats['iqr']:.2f}\n")
            
            f.write("\nФОРМА РАСПРЕДЕЛЕНИЯ:\n")
            f.write(f"  Skewness: {stats['skewness']:.3f}\n")
            if abs(stats['skewness']) < 0.5:
                f.write("    → Близко к симметричному\n")
            elif stats['skewness'] > 0.5:
                f.write("    → Скошено вправо (длинный хвост справа)\n")
            else:
                f.write("    → Скошено влево (длинный хвост слева)\n")
            
            f.write(f"  Kurtosis: {stats['kurtosis']:.3f}\n")
            if abs(stats['kurtosis']) < 1:
                f.write("    → Близко к нормальному\n")
            elif stats['kurtosis'] > 1:
                f.write("    → Острый пик (leptokurtic)\n")
            else:
                f.write("    → Плоский пик (platykurtic)\n")
            
            f.write("\nВЫБРОСЫ:\n")
            f.write(f"  Нижняя граница: {lower_bound:.2f}\n")
            f.write(f"  Верхняя граница: {upper_bound:.2f}\n")
            f.write(f"  Количество выбросов: {stats['outliers_count']:,}\n")
            f.write(f"  Процент выбросов: {stats['outliers_pct']:.1f}%\n")
            
            if 'suspicious_123' in stats:
                f.write(f"\n ПОДОЗРИТЕЛЬНЫЕ ЗНАЧЕНИЯ:\n")
                f.write(f"  Значений = 123: {stats['suspicious_123']}\n")
                f.write(f"  Возможно, это код для 'неизвестно'\n")
    
    # Сохраняем сводную таблицу
    stats_df = pd.DataFrame(numeric_stats)
    stats_df.to_csv(STEP2_DIR / 'numeric_features_statistics.csv', index=False)
    
    print(f"\n{Fore.GREEN} Анализ числовых признаков завершен!")
    print(f"   Результаты сохранены в: {numeric_dir}")
    
    return stats_df

# ==============================================================================
# ФУНКЦИЯ 4: АНАЛИЗ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ
# ==============================================================================

def analyze_categorical_features(train_df, categorical_features):
    """
    Детальный анализ категориальных признаков.
    
    Что анализируем:
    - Распределение категорий
    - Редкие категории
    - Связь с целевой переменной (approval rate по категориям)
    """
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{Fore.CYAN}📝 АНАЛИЗ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ")
    print(f"{Fore.CYAN}{'='*80}")
    
    # Создаем папку для категориальных признаков
    categorical_dir = STEP2_DIR / 'categorical_features'
    categorical_dir.mkdir(parents=True, exist_ok=True)
    
    categorical_stats = []
    
    for feature in categorical_features:
        print(f"\n Анализирую {feature}...")
        
        # Создаем папку для признака
        feature_dir = categorical_dir / feature
        feature_dir.mkdir(parents=True, exist_ok=True)
        
        # Считаем статистики
        value_counts = train_df[feature].value_counts()
        value_pcts = train_df[feature].value_counts(normalize=True) * 100
        
        stats = {
            'feature': feature,
            'unique_values': train_df[feature].nunique(),
            'missing': train_df[feature].isnull().sum(),
            'missing_pct': train_df[feature].isnull().sum() / len(train_df) * 100,
            'most_common': value_counts.index[0],
            'most_common_count': value_counts.values[0],
            'most_common_pct': value_pcts.values[0],
            'least_common': value_counts.index[-1],
            'least_common_count': value_counts.values[-1],
            'least_common_pct': value_pcts.values[-1]
        }
        
        categorical_stats.append(stats)
        
        # ========== АНАЛИЗ СВЯЗИ С ЦЕЛЕВОЙ ПЕРЕМЕННОЙ ==========
        # Считаем approval rate для каждой категории
        approval_by_category = train_df.groupby(feature)['loan_status'].agg(['mean', 'count'])
        approval_by_category.columns = ['approval_rate', 'count']
        approval_by_category['approval_rate'] *= 100  # В проценты
        
        # ========== ВИЗУАЛИЗАЦИЯ ==========
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # График 1: Распределение категорий
        value_counts.plot(kind='bar', ax=axes[0, 0], color='skyblue', edgecolor='black')
        axes[0, 0].set_title(f'{feature} - Distribution')
        axes[0, 0].set_xlabel(feature)
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Поворачиваем подписи если их много
        if len(value_counts) > 3:
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # График 2: Процентное распределение (pie chart)
        if len(value_counts) <= 10:  # Pie chart только если не слишком много категорий
            axes[0, 1].pie(value_counts.values, labels=value_counts.index, 
                          autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title(f'{feature} - Percentage Distribution')
        else:
            # Если категорий много, показываем топ-10
            top10 = value_counts.head(10)
            other = value_counts[10:].sum()
            if other > 0:
                plot_data = pd.concat([top10, pd.Series({'Other': other})])
            else:
                plot_data = top10
            axes[0, 1].pie(plot_data.values, labels=plot_data.index, 
                          autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title(f'{feature} - Top 10 Categories')
        
        # График 3: Approval rate по категориям
        approval_by_category['approval_rate'].plot(kind='bar', ax=axes[1, 0], 
                                                   color='green', edgecolor='black', alpha=0.7)
        axes[1, 0].axhline(y=train_df['loan_status'].mean() * 100, 
                          color='red', linestyle='--', 
                          label=f'Overall: {train_df["loan_status"].mean()*100:.1f}%')
        axes[1, 0].set_title(f'{feature} - Approval Rate by Category')
        axes[1, 0].set_xlabel(feature)
        axes[1, 0].set_ylabel('Approval Rate (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        if len(approval_by_category) > 3:
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # График 4: Stacked bar chart (approved vs rejected)
        crosstab = pd.crosstab(train_df[feature], train_df['loan_status'], normalize='index') * 100
        crosstab.plot(kind='bar', stacked=True, ax=axes[1, 1], 
                     color=['red', 'green'], alpha=0.7, edgecolor='black')
        axes[1, 1].set_title(f'{feature} - Approved vs Rejected')
        axes[1, 1].set_xlabel(feature)
        axes[1, 1].set_ylabel('Percentage')
        axes[1, 1].legend(['Rejected', 'Approved'])
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        if len(crosstab) > 3:
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        fig.suptitle(f'Детальный анализ: {feature}', fontsize=14, y=1.02)
        plt.tight_layout()
        
        # Сохраняем
        plt.savefig(feature_dir / f'{feature}_analysis.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # ========== СОХРАНЯЕМ ДЕТАЛЬНУЮ СТАТИСТИКУ ==========
        with open(feature_dir / f'{feature}_stats.txt', 'w') as f:
            f.write(f"Статистика для признака: {feature}\n")
            f.write("="*50 + "\n\n")
            
            f.write("ОСНОВНАЯ ИНФОРМАЦИЯ:\n")
            f.write(f"  Уникальных значений: {stats['unique_values']}\n")
            f.write(f"  Пропуски: {stats['missing']} ({stats['missing_pct']:.1f}%)\n")
            f.write(f"\nСАМОЕ ЧАСТОЕ ЗНАЧЕНИЕ:\n")
            f.write(f"  {stats['most_common']}: {stats['most_common_count']:,} ({stats['most_common_pct']:.1f}%)\n")
            f.write(f"\nСАМОЕ РЕДКОЕ ЗНАЧЕНИЕ:\n")
            f.write(f"  {stats['least_common']}: {stats['least_common_count']:,} ({stats['least_common_pct']:.1f}%)\n")
            
            f.write("\nРАСПРЕДЕЛЕНИЕ ЗНАЧЕНИЙ:\n")
            for val, count in value_counts.items():
                pct = count / len(train_df) * 100
                f.write(f"  {val:20}: {count:6,} ({pct:5.1f}%)\n")
            
            f.write("\nAPPROVAL RATE ПО КАТЕГОРИЯМ:\n")
            for idx, row in approval_by_category.iterrows():
                f.write(f"  {idx:20}: {row['approval_rate']:5.1f}% (n={row['count']:,})\n")
            
            # Находим категории с самым высоким и низким approval rate
            best_category = approval_by_category['approval_rate'].idxmax()
            worst_category = approval_by_category['approval_rate'].idxmin()
            
            f.write(f"\n Лучший approval rate: {best_category} ({approval_by_category.loc[best_category, 'approval_rate']:.1f}%)\n")
            f.write(f" Худший approval rate: {worst_category} ({approval_by_category.loc[worst_category, 'approval_rate']:.1f}%)\n")
    
    # Сохраняем сводную таблицу
    cat_stats_df = pd.DataFrame(categorical_stats)
    cat_stats_df.to_csv(STEP2_DIR / 'categorical_features_statistics.csv', index=False)

    print(f"\n{Fore.GREEN} Анализ категориальных признаков завершен!")
    print(f"   Результаты сохранены в: {categorical_dir}")
    
    return cat_stats_df

# ==============================================================================
# ФУНКЦИЯ 5: АНАЛИЗ СВЯЗЕЙ МЕЖДУ ПРИЗНАКАМИ
# ==============================================================================

def analyze_feature_relationships(train_df, features_dict):
    """
    Анализирует связи между признаками и с целевой переменной.
    
    Что ищем:
    - Корреляции между числовыми признаками
    - Сильные предикторы для loan_status
    - Мультиколлинеарность
    """
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{Fore.CYAN}🔗 АНАЛИЗ СВЯЗЕЙ МЕЖДУ ПРИЗНАКАМИ")
    print(f"{Fore.CYAN}{'='*80}")
    
    numeric_features = features_dict['numeric']
    
    # Создаем корреляционную матрицу
    corr_matrix = train_df[numeric_features + ['loan_status']].corr()
    
    # Корреляции с целевой переменной
    target_corr = corr_matrix['loan_status'].drop('loan_status').sort_values(ascending=False)
    
    print(f"\n🎯 Корреляции с loan_status:")
    print("-"*50)
    for feature, corr in target_corr.items():
        if abs(corr) > 0.1:
            strength = " СИЛЬНАЯ" if abs(corr) > 0.3 else "⚡ СРЕДНЯЯ" if abs(corr) > 0.2 else "💨 СЛАБАЯ"
            print(f"  {feature:30}: {corr:+.4f} {strength}")
    
    # Визуализация корреляционной матрицы
    plt.figure(figsize=(12, 10))
    
    # Создаем маску для верхнего треугольника
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Heatmap
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    
    plt.title('Корреляционная матрица признаков', fontsize=14, pad=20)
    plt.tight_layout()
    
    # Сохраняем
    plt.savefig(STEP2_DIR / 'correlation_matrix.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # Ищем пары с высокой корреляцией (мультиколлинеарность)
    high_corr_pairs = []
    for i in range(len(numeric_features)):
        for j in range(i+1, len(numeric_features)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > 0.7:
                high_corr_pairs.append({
                    'feature1': numeric_features[i],
                    'feature2': numeric_features[j],
                    'correlation': corr_value
                })
    
    if high_corr_pairs:
        print(f"\n{Fore.YELLOW}⚠️ Найдены пары с высокой корреляцией (>0.7):")
        for pair in high_corr_pairs:
            print(f"  {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")
        print(f"  → Возможна мультиколлинеарность, рассмотрите удаление одного из признаков")
    else:
        print(f"\n{Fore.GREEN} Высоких корреляций между признаками не найдено")
    
    # Сохраняем результаты
    target_corr.to_csv(STEP2_DIR / 'correlations_with_target.csv')
    
    return corr_matrix, target_corr

# ==============================================================================
# ФУНКЦИЯ 6: СОЗДАНИЕ ИТОГОВОГО ОТЧЕТА
# ==============================================================================

def create_summary_report(numeric_stats_df, categorical_stats_df, target_corr):
    """
    Создает итоговый отчет по результатам глубокого анализа.
    
    Собирает все ключевые находки в один файл.
    """
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{Fore.CYAN} СОЗДАНИЕ ИТОГОВОГО ОТЧЕТА")
    print(f"{Fore.CYAN}{'='*80}")
    
    with open(STEP2_DIR / 'SUMMARY_REPORT.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write(" "*20 + "LOAN APPROVAL PREDICTION\n")
        f.write(" "*15 + "Шаг 2: Детальный анализ признаков\n")
        f.write("="*70 + "\n\n")
        
        f.write("КЛЮЧЕВЫЕ НАХОДКИ:\n")
        f.write("-"*50 + "\n\n")
        
        # Проблемные признаки
        f.write("1. ПРОБЛЕМНЫЕ ПРИЗНАКИ:\n")
        
        # Проверяем на значения 123
        age_123 = numeric_stats_df[numeric_stats_df['feature'] == 'person_age']['suspicious_123'].values
        emp_123 = numeric_stats_df[numeric_stats_df['feature'] == 'person_emp_length']['suspicious_123'].values
        
        if len(age_123) > 0 and age_123[0] > 0:
            f.write(f"   • person_age: {int(age_123[0])} значений = 123 (аномалия)\n")
        if len(emp_123) > 0 and emp_123[0] > 0:
            f.write(f"   • person_emp_length: {int(emp_123[0])} значений = 123 (аномалия)\n")
        
        # Признаки с высоким % выбросов
        high_outliers = numeric_stats_df[numeric_stats_df['outliers_pct'] > 5]
        if not high_outliers.empty:
            f.write(f"\n   Признаки с большим количеством выбросов (>5%):\n")
            for _, row in high_outliers.iterrows():
                f.write(f"   • {row['feature']}: {row['outliers_pct']:.1f}% выбросов\n")
        
        # Топ предикторы
        f.write("\n2. САМЫЕ ВАЖНЫЕ ПРИЗНАКИ (по корреляции с loan_status):\n")
        for i, (feature, corr) in enumerate(target_corr.head(5).items(), 1):
            f.write(f"   {i}. {feature}: {corr:+.4f}\n")
        
        # Распределения
        f.write("\n3. ОСОБЕННОСТИ РАСПРЕДЕЛЕНИЙ:\n")
        
        # Скошенные распределения
        highly_skewed = numeric_stats_df[abs(numeric_stats_df['skewness']) > 2]
        if not highly_skewed.empty:
            f.write("   Сильно скошенные признаки (|skew| > 2):\n")
            for _, row in highly_skewed.iterrows():
                f.write(f"   • {row['feature']}: skewness = {row['skewness']:.2f}\n")
        
        # Категориальные признаки
        f.write("\n4. КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ:\n")
        for _, row in categorical_stats_df.iterrows():
            f.write(f"   • {row['feature']}: {row['unique_values']} категорий\n")
        
        # Рекомендации
        f.write("\n5. РЕКОМЕНДАЦИИ ДЛЯ PREPROCESSING:\n")
        f.write("   • Обработать значения 123 в person_age и person_emp_length\n")
        f.write("   • Применить log-трансформацию к person_income (сильно скошен)\n")
        f.write("   • One-hot encoding для категориальных признаков\n")
        f.write("   • Ordinal encoding для loan_grade (A→G порядок важен)\n")
        f.write("   • Рассмотреть создание новых признаков (ratios, interactions)\n")
        f.write("   • Использовать class_weight='balanced' из-за дисбаланса (14% approval)\n")
    
    print(f"{Fore.GREEN} Итоговый отчет создан: {STEP2_DIR / 'SUMMARY_REPORT.txt'}")

# ==============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ==============================================================================

def main():
    """
    Главная функция, которая запускает весь анализ.
    """
    print(f"{Fore.MAGENTA}{'='*80}")
    print(f"{Fore.MAGENTA}{' '*25} LOAN APPROVAL PREDICTION")
    print(f"{Fore.MAGENTA}{' '*22}Шаг 2: Детальный анализ")
    print(f"{Fore.MAGENTA}{'='*80}")
    
    # 1. Загружаем данные
    train_df, test_df = load_data()
    
    # 2. Классифицируем признаки
    features_dict = classify_features(train_df)
    
    # 3. Анализируем числовые признаки
    numeric_stats_df = analyze_numeric_features(train_df, features_dict['numeric'])
    
    # 4. Анализируем категориальные признаки
    categorical_stats_df = analyze_categorical_features(train_df, features_dict['categorical'])
    
    # 5. Анализируем связи между признаками
    corr_matrix, target_corr = analyze_feature_relationships(train_df, features_dict)
    
    # 6. Создаем итоговый отчет
    create_summary_report(numeric_stats_df, categorical_stats_df, target_corr)
    
    print(f"\n{Fore.MAGENTA}{'='*80}")
    print(f"{Fore.MAGENTA}{' '*30} ШАГ 2 ЗАВЕРШЕН!")
    print(f"{Fore.MAGENTA}{'='*80}")
    
    print(f"\n{Fore.YELLOW} Результаты сохранены в:")
    print(f"   {STEP2_DIR}")
    
    print(f"\n{Fore.YELLOW} Следующий шаг:")
    print(f"   Запустите: {Fore.CYAN}python src/01_eda/step3_check_quality.py")
    print(f"   Для проверки качества данных и обработки проблем")
    
    return train_df, test_df, features_dict

# ==============================================================================
# ТОЧКА ВХОДА
# ==============================================================================

if __name__ == "__main__":
    # Запускаем анализ
    train_df, test_df, features_dict = main()