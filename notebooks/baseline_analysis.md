# Baseline Model Analysis: Issues & Understanding

## 📊 Current Metrics

```
Accuracy: 64%
Macro Avg Recall: 55%
Weighted Avg F1-Score: 0.56
```

### Classification Report

```
              precision    recall  f1-score   support

          -1       1.00      0.00      0.01      1650
           0       0.68      0.80      0.73      2555
           1       0.62      0.85      0.72      3154

    accuracy                           0.64      7359
   macro avg       0.77      0.55      0.49      7359
weighted avg       0.72      0.64      0.56      7359
```

---

## 🎯 The Core Problem: Severe Class Imbalance

### Class Distribution

- **-1 (Negative)**: 1,650 samples (22.4%) ← **Minority class**
- **0 (Neutral)**: 2,555 samples (34.7%)
- **1 (Positive)**: 3,154 samples (42.9%) ← **Majority class**

**Imbalance Ratio:** 3154 / 1650 = **1.91x difference**

The model sees positive comments ~2x more often than negative ones during training, creating a bias.

---

## 🚨 Class -1 (Negative Comments) Crisis

| Metric | Value | Status | Meaning |
|--------|-------|--------|---------|
| **Precision** | 1.00 | ✅ Perfect | When model predicts -1, it's always right |
| **Recall** | 0.00 | ❌ Terrible | Model NEVER predicts -1 despite 1,650 existing samples |
| **F1-Score** | 0.01 | 💔 Useless | Effectively broken for this class |
| **Support** | 1,650 | — | 1,650 negative comments exist, all missed |

### What's Actually Happening

The model has learned a dangerous strategy:
1. **During training:** Negative comments are rare (22.4% vs 43% positive)
2. **Model thinks:** "Predicting positive/neutral = higher overall accuracy"
3. **Avoidance strategy:** Model learns to NEVER predict -1 to avoid the penalty
4. **Result:** 100% recall failure on negative class

**Example scenario:**
- True label: -1 (negative comment)
- Model prediction: 1 (positive comment) ❌
- This happens for ~1,650 samples (100% of them)

---

## 📉 Other Classes (Fair Performance, Not Great)

### Class 0 (Neutral)
- **Precision:** 0.68 | **Recall:** 0.80
- Better balanced than -1, but still issues
- 32% false positives: predicting neutral when it's actually positive/negative

### Class 1 (Positive)
- **Precision:** 0.62 | **Recall:** 0.85
- High recall ✅ (catches most positives)
- Low precision ❌ (many false positives)
- Model is "too optimistic" about positivity

---

## ⚠️ Why 64% Accuracy is Misleading

A naive baseline: "Always predict class 1 (majority class)"
- Accuracy would be: 3,154 / 7,359 = **42.8%**
- Our model: **64%** accuracy

**Improvement: Only +21.2 percentage points** (not as impressive as it sounds)

The model isn't learning sentiment nuances—it's just predicting majority classes more often.

---

## 🎯 Key Insights

### 1. **Imbalanced Data Dominates Performance**
- Without addressing class imbalance, model is inherently biased
- Minority class (negative) gets sacrificed for overall accuracy

### 2. **Accuracy is the Wrong Metric**
- Should use: F1-score, Precision-Recall curves, Balanced Accuracy, Macro F1
- Accuracy hides poor minority class performance

### 3. **BoW + Random Forest Underfits Minority Class**
- Too simplistic to capture negative sentiment patterns
- Needs stronger feature representation or model complexity

### 4. **Model is Unstable for Minority Class**
- High precision but zero recall = model avoids predictions entirely
- This is worse than moderate performance on all classes

---

## 🔧 Solutions (Next Steps)

### Priority 1: Address Class Imbalance
- **Option A:** Add class weights to Random Forest
  ```python
  class_weight = {-1: 3, 0: 1.4, 1: 1}  # Weight minority class higher
  RandomForestClassifier(class_weight=class_weight, ...)
  ```

- **Option B:** Use SMOTE (Synthetic Minority Over-sampling Technique)
  ```python
  from imblearn.over_sampling import SMOTE
  smote = SMOTE()
  X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
  ```

### Priority 2: Better Evaluation Metrics
- Use **Macro F1-Score** (equal weight to all classes)
- Use **Balanced Accuracy** (average of per-class recalls)
- Monitor **Precision-Recall curves** per class

### Priority 3: Hyperparameter Optimization
- Use Optuna to find best params WITH class balancing
- Don't just optimize for accuracy—optimize for Macro F1 or Balanced Accuracy

### Priority 4: Model Architecture
- Try stronger models (Gradient Boosting, SVM)
- Try better features (TF-IDF, n-grams)
- Eventually: Transformer-based models (BERT-like)

---

## 📋 Baseline Summary

| Aspect | Status | Assessment |
|--------|--------|------------|
| **Overall Performance** | 64% accuracy | Moderate, but misleading |
| **Negative Class** | 0% recall | **CRITICAL FAILURE** |
| **Neutral/Positive Classes** | 62-85% recall | Acceptable, needs improvement |
| **Model Stability** | Imbalanced | Biased toward majority |
| **Ready for Production?** | ❌ No | Must fix class imbalance first |

---

## 🚀 Next Experiment Plan

**Experiment 2 (Planned):** Add class weights to baseline
- Same BoW + Random Forest
- But with `class_weight='balanced'`
- Expected: Better negative class recall, slight trade-off in overall accuracy

**Experiment 3 (Planned):** Try SMOTE resampling
- Oversample negative comments to match majority
- Should improve minority class without losing majority class

**Experiment 4+ (Planned):** Optuna optimization
- Sweep hyperparameters for best Macro F1-Score
- Try multiple vectorizers (TF-IDF, n-grams)
- Find optimal parameters across balanced/imbalanced settings

---

**Generated:** 2026-07-20 | **Model:** RandomForest Baseline (BoW, max_features=10000)
