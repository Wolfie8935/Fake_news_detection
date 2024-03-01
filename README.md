# Fake News Detection Model

**Objective:** The goal of this model is to classify news articles as either fake or real using machine learning techniques.

**Dataset:** The dataset used is "fake_or_real_news.csv," which contains news articles labeled as "FAKE" or "REAL."
## Preprocessing:

- The dataset was loaded into a Pandas DataFrame.
- A new binary column 'fake' was created, where "REAL" articles are labeled as 0 and "FAKE" articles are labeled as 1.
- The 'label' column was dropped, and the remaining data was split into features (x) and target labels (y).

## Feature Extraction:

- The TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer was used to convert the text data into numerical format, which is suitable for machine learning models. Stop words in English were removed, and the maximum document frequency was set to 0.7 to filter out too common words.

## Model Training:

- The data was split into training and testing sets, with 80% of the data used for training and 20% for testing.
- A Linear Support Vector Classifier (LinearSVC) was trained on the vectorized training data.

## Evaluation:

- The trained model achieved an accuracy of **94.08050513022889%** on the test set, indicating a high level of performance in distinguishing between fake and real news articles.

## Prediction:

- The model can be used to predict the authenticity of new news articles by vectorizing the text and applying the trained classifier.

## Graphical Representation:

To visualize the results, you can create a confusion matrix and a classification report:

```python
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

# Predictions on the test set
y_pred = clf.predict(x_test_vectorized)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
```
This code will generate a confusion matrix showing the number of true positives, true negatives, false positives, and false negatives. It also prints a classification report that includes precision, recall, and F1-score for both classes.
