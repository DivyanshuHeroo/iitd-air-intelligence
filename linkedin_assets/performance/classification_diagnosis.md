# Classification Diagnosis

## Current Performance
- **Exact 6-Class Accuracy**: 61.9% (Prior to improvements, was around 61.9%)
- **Macro F1**: ~0.31
- **Weighted F1**: ~0.56

## Target Definition
The target was the exact PM2.5 categories (Good, Satisfactory, Moderate, Poor, Very Poor, Severe) for the *next hour*.

## Leakage Check
- No direct target leakage was found; validation is strictly time-based (80% train, 20% test).
- Features rely only on historical data (lagged or shifted rolling stats).

## Class Distribution in Delhi-NCR (Winter)
- Good: 4
- Satisfactory: 2,056
- Moderate: 5,693
- Poor: 9,917
- Very Poor: 38,072
- Severe: 18,121

## Key Findings
1. **Severe Class Imbalance**: The vast majority of the data falls into "Very Poor" and "Severe". The "Good" class almost never occurs.
2. **Misleading Accuracy**: An accuracy of 61.9% is largely driven by predicting the majority class ("Very Poor"). The Macro F1 score of 0.31 indicates that the model struggles significantly with the minority classes.
3. **Confusion**: Most errors are adjacent class confusions (e.g., confusing Very Poor with Severe or Poor), which is natural since PM2.5 is a continuous variable being artificially discretized.
