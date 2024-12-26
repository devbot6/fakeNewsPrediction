# Fake News Predictor using Logistic Regression

This project utilizes machine learning to predict whether a news segment is fake or not. The model is trained on over 20,000 data points, including news titles, authors, and labels (fake or not). The logistic regression model is built using the `scikit-learn` library, and various techniques such as stemming, vectorization, and grid search are applied to improve the model's performance.

## Features

- **Data Preprocessing**: 
  - Replaces missing values with empty strings.
  - Merges columns (author and title) to create a content column.
  - Applies stemming to reduce words to their root form.
  - Vectorizes text data using `TfidfVectorizer`.

- **Modeling**:
  - Trains a logistic regression model using the training data.
  - Performs grid search to optimize hyperparameters.
  
- **Evaluation**:
  - Model accuracy is evaluated using confusion matrix, classification report, and ROC-AUC curve.
  
- **Visualization**:
  - **Confusion Matrix**: Displays the model’s performance.
  - **ROC Curve**: Plots true vs false positive rates.
  - **Word Cloud**: Shows word frequency in the training data.

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- nltk
- wordcloud

