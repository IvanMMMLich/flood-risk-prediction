"""
===============================================================================
                    STEP 7: FINAL MODEL & SUBMISSION
                    
                    Финальная модель и создание файла для отправки
===============================================================================

ЦЕЛЬ:
-----
1. Обучить финальную модель на ВСЕХ train данных
2. Попробовать XGBoost для еще лучшего результата
3. Подобрать оптимальные гиперпараметры
4. Создать submission файл для Kaggle
5. Сохранить финальную модель

ОЖИДАЕМЫЙ РЕЗУЛЬТАТ:
-------------------
- Финальная модель с ROC-AUC > 0.93
- submission.csv готовый к отправке
- Документация по использованию модели
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb 
import pickle
from datetime import datetime

warnings.filterwarnings('ignore')

# Пути
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_ENGINEERED = ROOT_DIR / 'data' / 'engineered'
DATA_RAW = ROOT_DIR / 'data' / 'raw'
RESULTS = ROOT_DIR / 'results' / 'final_model'
MODEL_V3 = ROOT_DIR / 'results' / 'model_versions' / 'v3_final'
SUBMISSIONS = ROOT_DIR / 'submissions'

# Создаем папки
RESULTS.mkdir(parents=True, exist_ok=True)
MODEL_V3.mkdir(parents=True, exist_ok=True)
SUBMISSIONS.mkdir(parents=True, exist_ok=True)

def load_data():
    """Загрузка engineered данных."""
    print("\n" + "="*60)
    print("STEP 7: FINAL MODEL & SUBMISSION")
    print("="*60)
    
    # Загружаем обработанные данные
    train_df = pd.read_csv(DATA_ENGINEERED / 'train_engineered.csv')
    test_df = pd.read_csv(DATA_ENGINEERED / 'test_engineered.csv')
    
    # Загружаем оригинальный test для ID
    test_original = pd.read_csv(DATA_RAW / 'test.csv')
    
    print(f"\n📊 Загружено:")
    print(f"   Train: {len(train_df)} записей, {len(train_df.columns)} признаков")
    print(f"   Test: {len(test_df)} записей")
    
    return train_df, test_df, test_original['id']

def train_random_forest_optimized(X_train, y_train, X_val, y_val):
    """Обучение Random Forest с подбором параметров."""
    print("\n🌲 RANDOM FOREST с оптимизацией:")
    print("-"*40)
    
    # Сначала быстрый поиск лучших параметров
    print("Подбор гиперпараметров (займет 1-2 минуты)...")
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [10, 20],
        'min_samples_leaf': [5, 10]
    }
    
    rf = RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    grid_search = GridSearchCV(
        rf, 
        param_grid, 
        cv=3, 
        scoring='roc_auc',
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n✅ Лучшие параметры:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param}: {value}")
    
    # Обучаем финальную модель с лучшими параметрами
    best_rf = grid_search.best_estimator_
    
    # Оценка на валидации
    y_pred_proba = best_rf.predict_proba(X_val)[:, 1]
    rf_auc = roc_auc_score(y_val, y_pred_proba)
    
    print(f"\n📊 Random Forest ROC-AUC: {rf_auc:.4f}")
    
    return best_rf, rf_auc

def train_xgboost(X_train, y_train, X_val, y_val):
    """Обучение XGBoost."""
    print("\n🚀 XGBOOST модель:")
    print("-"*40)
    
    # Преобразуем в DMatrix для XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Параметры XGBoost
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42,
        'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum()  # для дисбаланса
    }
    
    # Обучение с early stopping
    evals = [(dtrain, 'train'), (dval, 'val')]
    
    print("Обучение XGBoost...")
    xgb_model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    # Предсказание
    y_pred_proba = xgb_model.predict(dval)
    xgb_auc = roc_auc_score(y_val, y_pred_proba)
    
    print(f"\n📊 XGBoost ROC-AUC: {xgb_auc:.4f}")
    
    return xgb_model, xgb_auc

def train_final_model(X_train_full, y_train_full):
    """Обучение финальной модели на ВСЕХ данных."""
    print("\n" + "="*60)
    print("ФИНАЛЬНАЯ МОДЕЛЬ (на всех данных)")
    print("="*60)
    
    # Используем лучшие параметры из поиска
    final_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    print("\n🎯 Обучение финальной модели на ВСЕХ train данных...")
    final_model.fit(X_train_full, y_train_full)
    
    # Cross-validation для оценки
    print("\nCross-validation (5 folds)...")
    cv_scores = cross_val_score(
        final_model, X_train_full, y_train_full, 
        cv=5, scoring='roc_auc', n_jobs=-1
    )
    
    print(f"\nCross-validation ROC-AUC:")
    for i, score in enumerate(cv_scores, 1):
        print(f"   Fold {i}: {score:.4f}")
    print(f"   Среднее: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return final_model, cv_scores.mean()

def analyze_feature_importance(model, feature_names):
    """Анализ важности признаков финальной модели."""
    print("\n📊 ВАЖНОСТЬ ПРИЗНАКОВ (финальная модель):")
    print("-"*40)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Топ-15 признаков
    print("\nТоп-15 важнейших признаков:")
    for idx, row in importance_df.head(15).iterrows():
        bar = '█' * int(row['importance'] * 100)
        print(f"{row['feature']:30s}: {row['importance']:.3f} {bar}")
    
    # Визуализация
    plt.figure(figsize=(10, 8))
    top_20 = importance_df.head(20)
    
    plt.barh(range(len(top_20)), top_20['importance'], color='steelblue')
    plt.yticks(range(len(top_20)), top_20['feature'])
    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importance - Final Model')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(RESULTS / 'final_feature_importance.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    return importance_df

def create_submission(model, X_test, test_ids):
    """Создание файла submission."""
    print("\n📝 СОЗДАНИЕ SUBMISSION:")
    print("-"*40)
    
    # Предсказания
    if hasattr(model, 'predict'):  # Random Forest
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
    else:  # XGBoost
        dtest = xgb.DMatrix(X_test)
        probabilities = model.predict(dtest)
        predictions = (probabilities > 0.5).astype(int)
    
    # Создаем DataFrame
    submission = pd.DataFrame({
        'id': test_ids,
        'loan_status': predictions
    })
    
    # Статистика
    print(f"\n📊 Распределение предсказаний:")
    print(f"   Отказы (0): {(predictions == 0).sum()} ({(predictions == 0).mean():.1%})")
    print(f"   Одобрения (1): {(predictions == 1).sum()} ({(predictions == 1).mean():.1%})")
    
    # Проверка соотношения
    train_ratio = 0.142  # 14.2% одобрений в train
    pred_ratio = predictions.mean()
    print(f"\n   Train ratio: {train_ratio:.1%}")
    print(f"   Pred ratio: {pred_ratio:.1%}")
    print(f"   Разница: {abs(pred_ratio - train_ratio):.1%}")
    
    # Сохранение
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'submission_{timestamp}.csv'
    submission.to_csv(SUBMISSIONS / filename, index=False)
    
    # Также сохраняем как последний
    submission.to_csv(SUBMISSIONS / 'submission_latest.csv', index=False)
    
    print(f"\n✅ Submission сохранен:")
    print(f"   {SUBMISSIONS / filename}")
    print(f"   {SUBMISSIONS / 'submission_latest.csv'}")
    
    return submission

def save_final_model(model, cv_score, feature_importance):
    """Сохранение финальной модели и метаданных."""
    print("\n💾 Сохранение финальной модели...")
    
    # Сохраняем модель
    with open(MODEL_V3 / 'model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Сохраняем метрики
    with open(MODEL_V3 / 'metrics.txt', 'w') as f:
        f.write("FINAL MODEL METRICS\n")
        f.write("="*50 + "\n")
        f.write(f"Model: RandomForestClassifier\n")
        f.write(f"Cross-validation ROC-AUC: {cv_score:.4f}\n")
        f.write(f"Number of features: {len(feature_importance)}\n")
        f.write(f"Training date: {datetime.now()}\n")
        f.write("\nTop 10 features:\n")
        for idx, row in feature_importance.head(10).iterrows():
            f.write(f"  {row['feature']}: {row['importance']:.3f}\n")
    
    # Сохраняем важность признаков
    feature_importance.to_csv(MODEL_V3 / 'feature_importance.csv', index=False)
    
    print(f"✅ Модель сохранена в {MODEL_V3}")

def main():
    """Основная функция."""
    
    # Загрузка данных
    train_df, test_df, test_ids = load_data()
    
    # Подготовка данных
    X_train_full = train_df.drop(['loan_status'], axis=1, errors='ignore')
    y_train_full = train_df['loan_status']
    X_test = test_df
    
    # Разделение для валидации (для сравнения моделей)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_train_full
    )
    
    # Обучение разных моделей
    rf_model, rf_auc = train_random_forest_optimized(X_train, y_train, X_val, y_val)
    
    # XGBoost (опционально - нужно установить: pip install xgboost)
    try:
        xgb_model, xgb_auc = train_xgboost(X_train, y_train, X_val, y_val)
    except:
        print("\n⚠️ XGBoost не установлен. Используем Random Forest.")
        xgb_model, xgb_auc = None, 0
    
    # Выбор лучшей модели
    if xgb_auc > rf_auc:
        print(f"\n🏆 Лучшая модель: XGBoost (AUC={xgb_auc:.4f})")
        best_model_type = 'xgboost'
    else:
        print(f"\n🏆 Лучшая модель: Random Forest (AUC={rf_auc:.4f})")
        best_model_type = 'rf'
    
    # Обучение финальной модели на ВСЕХ данных
    final_model, cv_score = train_final_model(X_train_full, y_train_full)
    
    # Анализ важности
    feature_importance = analyze_feature_importance(final_model, X_train_full.columns)
    
    # Создание submission
    submission = create_submission(final_model, X_test, test_ids)
    
    # Сохранение
    save_final_model(final_model, cv_score, feature_importance)
    
    # Итоговый отчет
    print("\n" + "="*60)
    print("🎉 ПРОЕКТ ЗАВЕРШЕН!")
    print("="*60)
    print(f"✅ Финальная модель обучена")
    print(f"✅ Cross-validation AUC: {cv_score:.4f}")
    print(f"✅ Submission создан: {len(submission)} предсказаний")
    print(f"✅ Все результаты сохранены")
    print("\n🚀 Готово к отправке на Kaggle!")

if __name__ == "__main__":
    main()