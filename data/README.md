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
