# Credit-Card-Fraud-Detection

Within this project, we will employ diverse predictive models to assess their precision in distinguishing between regular payments and fraudulent transactions. The dataset specifies that the features are scaled and their names are withheld for privacy considerations. Nonetheless, we can still examine crucial facets of the dataset.

### Our Objectives:
Comprehend the limited distribution of the "little" data provided to us.
Establish a 50/50 sub-dataframe ratio for "Fraud" and "Non-Fraud" transactions using the NearMiss Algorithm.
Identify the classifiers to be employed and determine the one with higher accuracy.
Construct a Neural Network and compare its accuracy with our best classifier.
Grasp common mistakes associated with imbalanced datasets.

### Outline:
I. Understanding our data
a) Obtaining a sense of our data

II. Preprocessing
a) Scaling and Distributing
b) Data Splitting

III. Random UnderSampling and Oversampling
a) Distributing and Correlating
b) Anomaly Detection
c) Dimensionality Reduction and Clustering (t-SNE)
d) Classifiers
e) In-Depth Exploration of Logistic Regression
f) Oversampling with SMOTE

IV. Testing
a) Testing with Logistic Regression
b) Neural Networks Testing (Undersampling vs. Oversampling)

### References:

"Hands on Machine Learning with Scikit-Learn & TensorFlow" by Aurélien Géron (O'Reilly), Copyright 2017 Aurélien Géron.
"Machine Learning - Over-& Undersampling - Python/Scikit/Scikit-Imblearn" by Coding-Maniac.
"auprc, 5-fold c-v, and resampling methods" by Jeremy Lane (Kaggle Notebook).
Gathering a Sense of Our Data:
Initially, we need to obtain a basic understanding of our data. Keep in mind that, except for the transaction and amount, we do not know the nature of the other columns due to privacy reasons. The only information available is that those undisclosed columns have undergone scaling.

### Summary:

The transaction amount is relatively small, with an approximate mean of USD 88.
There are no "Null" values, eliminating the need for value replacement strategies.
Most transactions were Non-Fraudulent (99.83%), while Fraudulent transactions occurred 0.17% of the time in the dataframe.
Feature Technicalities:

PCA Transformation: The data description mentions that all features underwent PCA transformation (Dimensionality Reduction technique), except for time and amount.
Scaling: It's important to note that to implement a PCA transformation, features must be pre-scaled. In this case, all the V features are assumed to have been scaled, based on the dataset developers' actions.



### Notes :

Never test on the oversampled or undersampled dataset.
If implementing cross-validation, oversample or undersample the training data during cross-validation, not before!
Avoid using accuracy score as a metric with imbalanced datasets (as it can be misleading); instead, use f1-score, precision/recall score, or confusion matrix.
