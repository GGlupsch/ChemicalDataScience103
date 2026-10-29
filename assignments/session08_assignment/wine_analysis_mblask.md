# Exploratory Data Analysis of the Wine Dataset

## Dataset and initial inspection

The Wine dataset contains 178 observations, 13 numerical chemical features and three wine classes (59, 71 and 48 samples, respectively). No missing values or duplicate rows were found.

The features differ considerably in scale, with proline ranging from 278 to 1680 and hue from 0.48 to 1.71. Standardisation was therefore applied before PCA and logistic regression to prevent features with larger numerical values from dominating.

## Distributions and differences between classes

The pairplot shows differences between the wine classes, although individual features generally do not separate them completely. The flavanoid boxplot illustrates this: class 2 has substantially lower mean flavanoid content (0.78) than classes 0 (2.98) and 1 (2.08), which overlap.

Class 0 has the highest mean proline content (1115.71), while class 2 has the highest mean colour intensity (7.40). Some observations lie towards the extremes of their distributions, particularly for colour intensity and malic acid, but these do not necessarily indicate measurement errors. Overall, combining several features appears more useful for distinguishing the classes than relying on individual measurements.

## Correlations

The correlation heatmap shows strong positive correlations between total phenols and flavanoids (*r* = 0.865), flavanoids and OD280/OD315 (*r* = 0.787), and alcohol and proline (*r* = 0.644). Malic acid and hue are negatively correlated (*r* = -0.561).

These relationships indicate that some features contain overlapping information, particularly the phenolic measurements. However, correlations do not establish causality and may partly reflect differences between wine classes.

## Principal component analysis

PCA was performed on the 13 standardised features. PC1 explains 36.20% and PC2 19.21% of the total variance, giving a combined 55.41%.

PC1 is mainly associated with flavanoids, total phenols and OD280/OD315, while PC2 is associated with colour intensity, alcohol and proline. The PCA scatterplot shows partial separation of the three classes, with some overlap remaining.

Thus, the first two components capture much of the class structure despite retaining only slightly more than half of the total variance. However, PCA maximises variance rather than class separation, so the components are not necessarily optimal for classification.

## Classification

The dataset was divided into stratified training (142 samples) and test (36 samples) sets using `random_state=42`. Logistic regression, a decision tree and a random forest were compared using all 13 features and the first two principal components. Scaling and PCA for classification were fitted on training data only.

| Model | 13 features | 2 PCs |
|---|---:|---:|
| Logistic regression | 97.2% (35/36) | 91.7% (33/36) |
| Decision tree | 94.4% (34/36) | 88.9% (32/36) |
| Random forest | 100.0% (36/36) | 88.9% (32/36) |

All three classifiers performed well with the original features, with the random forest achieving the highest accuracy. Performance decreased when using only two PCs, indicating that some information relevant to classification was lost.

The decision tree assigns the greatest feature importance to flavanoids (0.408) and colour intensity (0.400). Both also rank highly in the random forest (0.160 and 0.188, respectively), alongside proline (0.147). These findings are consistent with the exploratory analysis, although feature importance does not imply causality.

Since the results are based on one split and only 36 test samples, the differences in accuracy should be interpreted cautiously. Repeated cross-validation would provide a more reliable comparison.

*Note:* Logistic regression was used instead of the linear regression mentioned in the README, following the Python template and the categorical nature of the target.

## Possible research questions and hypotheses

1. **Can phenolic properties distinguish class 2?** Its lower flavanoid content suggests that a model using only phenolic features might already achieve good classification accuracy.
2. **How important is proline for distinguishing class 0?** Its higher mean concentration and feature importance suggest that removing it could reduce classification performance.
3. **Does the random forest consistently outperform the other models?** Repeated stratified cross-validation could establish whether its advantage persists across different splits.

Overall, combining chemical features allows the three wine classes to be distinguished with rather high accuracy. PCA provides a useful visualisation of their structure, although the original features gave better classification results.
