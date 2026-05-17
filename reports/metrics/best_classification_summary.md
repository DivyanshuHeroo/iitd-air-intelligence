# Best Classification Summary

## Exact 6-Class Prediction (Next Hour)
- **Best Model**: RandomForest
- **Accuracy**: 59.27%
- **Top-2 Accuracy**: 60.96%
- **Adjacent Accuracy**: 95.59%

## 3-Class Severity Prediction (Next Hour)
- **Best Model**: VotingClassifier
- **Accuracy**: 84.90%

## Binary Unsafe-Air Prediction (Next Hour)
- **Best Model**: ExtraTrees
- **Accuracy**: 99.44%

## Did we achieve 85%? 
An accuracy of 85%+ was NOT achieved for the exact 6-class prediction. The best model reached ~59%. However, predicting the exact class is highly noisy due to arbitrary boundaries. When allowing for adjacent-class correctness, the accuracy jumps to ~95%, showing the model captures the general severity trend extremely well. For broader severity classification (3-class), the accuracy reached ~85%. For binary (Acceptable vs Unsafe), it reached ~99%, though this is heavily driven by class imbalance since most hours in Delhi-NCR winter are 'Unsafe'.