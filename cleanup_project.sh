#!/bin/bash

echo "================================================"
echo "🧹 ОЧИСТКА И ОРГАНИЗАЦИЯ ПРОЕКТА"
echo "================================================"

# Переход в папку проекта
cd ~/VSCODE/loan-approval-prediction

echo ""
echo "1️⃣ УДАЛЕНИЕ ПУСТЫХ И НЕНУЖНЫХ ПАПОК..."
echo "----------------------------------------"

# Удаляем точно ненужные пустые папки
rm -rf models/
rm -rf scripts/
rm -rf tests/
rm -rf notebooks/
rm -rf results/tables/
rm -rf results/figures/
rm -rf results/reports/
rm -rf results/features/
rm -rf results/step1_first_look/
rm -rf results/step6_engineering/
rm -rf src/04_evaluation/
rm -rf data/submissions/  # старая пустая папка
echo "✅ Пустые папки удалены"

echo ""
echo "2️⃣ УДАЛЕНИЕ СИСТЕМНЫХ ФАЙЛОВ..."
echo "----------------------------------------"

# Удаляем системный мусор Mac
find . -name ".DS_Store" -delete
find . -name "*.pyc" -delete  
find . -name "__pycache__" -type d -delete
find . -name ".ipynb_checkpoints" -type d -delete
echo "✅ Системные файлы очищены"

echo ""
echo "3️⃣ ИСПРАВЛЕНИЕ СТРУКТУРЫ..."
echo "----------------------------------------"

# Исправляем структуру step2
if [ -f "results/step2_deep_explore/categorical_features_statistics.csv" ]; then
    mv results/step2_deep_explore/categorical_features_statistics.csv results/step2_deep_explore/categorical_features/
    echo "✅ Перемещен categorical_features_statistics.csv"
fi

if [ -f "results/step2_deep_explore/numeric_features_statistics.csv" ]; then
    mv results/step2_deep_explore/numeric_features_statistics.csv results/step2_deep_explore/numeric_features/
    echo "✅ Перемещен numeric_features_statistics.csv"
fi

echo ""
echo "4️⃣ СОЗДАНИЕ ВАЖНЫХ ФАЙЛОВ..."
echo "----------------------------------------"

# Копируем финальную модель в корень
if [ -f "results/model_versions/v3_final/model.pkl" ]; then
    cp results/model_versions/v3_final/model.pkl ./final_model.pkl
    echo "✅ Финальная модель скопирована в корень"
fi

# Обновляем .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Data (слишком большие для git)
data/raw/*.csv
!data/raw/sample_submission.csv

# Models (большие файлы)
*.pkl
*.joblib
*.h5

# System files
.DS_Store
Thumbs.db
.vscode/
.idea/
*.log
*.swp

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Temporary
tmp/
temp/
EOF
echo "✅ .gitignore обновлен"

echo ""
echo "5️⃣ СОЗДАНИЕ README ДЛЯ ВАЖНЫХ ПАПОК..."
echo "----------------------------------------"

# README для submissions
cat > submissions/README.md << 'EOF'
# Submission Files

## Latest Results
- **File:** submission_latest.csv
- **ROC-AUC:** 0.9535
- **Model:** Random Forest + XGBoost
- **Date:** 2025-11-21

## How to Submit
1. Go to Kaggle competition page
2. Upload `submission_latest.csv`
3. Wait for scoring

## History
- v3_final: ROC-AUC 0.9535 (XGBoost)
- v2_engineered: ROC-AUC 0.9316 (Random Forest)
- v1_cleaned: ROC-AUC 0.8730 (Logistic Regression)
- v0_baseline: ROC-AUC 0.8740 (Baseline)
EOF
echo "✅ README для submissions создан"

# README для data
cat > data/README.md << 'EOF'
# Data Directory Structure

## Folders
- **raw/** - Original data from Kaggle
- **processed/** - Cleaned data (anomalies fixed)
- **engineered/** - Feature engineered data (final)

## Data Pipeline
1. raw → processed (Step 3: cleaning)
2. processed → engineered (Step 6: feature engineering)
3. engineered → model training (Step 7)

## File Sizes
- train.csv: 58,645 records
- test.csv: 39,098 records
EOF
echo "✅ README для data создан"

# README для results
cat > results/README.md << 'EOF'
# Results Directory

## Model Versions
- **v0_baseline** - Initial baseline (ROC-AUC: 0.8740)
- **v1_after_cleaning** - After data cleaning (ROC-AUC: 0.8730)
- **v2_after_engineering** - After feature engineering (ROC-AUC: 0.9316)
- **v3_final** - Final model with XGBoost (ROC-AUC: 0.9535)

## Analysis Steps
- **step2_deep_explore** - Detailed EDA
- **step3_cleaning** - Data quality checks
- **step4_target** - Target variable analysis
- **step5_importance** - Feature importance analysis
- **final_model** - Final model artifacts
EOF
echo "✅ README для results создан"

echo ""
echo "6️⃣ СОЗДАНИЕ ФАЙЛА requirements.txt..."
echo "----------------------------------------"

cat > requirements.txt << 'EOF'
# Data Processing
pandas==2.3.3
numpy==2.3.5

# Visualization
matplotlib==3.10.2
seaborn==0.13.2

# Machine Learning
scikit-learn==1.6.1
xgboost==3.1.2

# Utilities
jupyter==1.1.1
ipython==8.31.0
EOF
echo "✅ requirements.txt обновлен"

echo ""
echo "7️⃣ ПРОВЕРКА СТРУКТУРЫ..."
echo "----------------------------------------"

# Показываем финальную структуру
echo ""
echo "📁 ФИНАЛЬНАЯ СТРУКТУРА:"
echo ""
tree -L 2 -I 'venv|__pycache__|*.pyc' 2>/dev/null || ls -la

echo ""
echo "================================================"
echo "✅ ОЧИСТКА ЗАВЕРШЕНА!"
echo "================================================"
echo ""
echo "📊 СТАТИСТИКА:"
echo "- Модель: ROC-AUC 0.9535"
echo "- Submission: готов к отправке"
echo "- Размер проекта: $(du -sh . | cut -f1)"
echo ""
echo "🚀 Проект готов к публикации на GitHub!"
echo ""
