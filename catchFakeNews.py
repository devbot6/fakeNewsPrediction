import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import nltk
nltk.download('stopwords')

train_path = 'C:/Users/devhu/Desktop/codeProjects/fakeNews/train.csv'
test_path = 'C:/Users/devhu/Desktop/codeProjects/fakeNews/test.csv'
submit_path = 'C:/Users/devhu/Desktop/codeProjects/fakeNews/submit.csv'

print(stopwords.words('english'))

train_df = pd.read_csv(train_path)
print(train_df.shape)

print(train_df.head(15))

# get the number of missing values in the training dataset
train_df.isnull().sum()

# replacing the null values with empty string
train_df = train_df.fillna('')

# merging the author name and news title
train_df['content'] = train_df['author']+' '+ train_df['title']

train_df.info()

# separating the data & label
X = train_df.drop(columns='label', axis=1)
y = train_df['label']

print(X)
print(y)

def apply_stemming(content, stemmer):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [stemmer.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

stemmer = PorterStemmer()
train_df['content'] = train_df['content'].apply(lambda content: apply_stemming(content, stemmer))

print(train_df['content'])

#separating the features (stemmed contnent) and label
X = train_df['content'].values
y = train_df['label'].values

X.shape
y.shape



# convert text data to its numerical representation
vectorizer = TfidfVectorizer(max_features=9000)
vectorizer.fit(X)

X = vectorizer.transform(X)
X.shape



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Find optimal hyperparameters using grid search
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter
    'penalty': ['l1', 'l2']  # Regularization type
}

# Create a Grid Search with cross-validation
grid_search = GridSearchCV(LogisticRegression(solver='liblinear'), param_grid, cv=5, scoring='accuracy')

# Fit the Grid Search to the training data
grid_search.fit(X_train, y_train)

cv_results = pd.DataFrame(grid_search.cv_results_)
print(cv_results[['mean_test_score', 'std_test_score', 'params']])

# Get the best hyperparameters
best_params = grid_search.best_params_

# Train a Logistic Regression model with the best hyperparameters
best_logistic_regression = LogisticRegression(solver='liblinear', **best_params)
best_logistic_regression.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_logistic_regression.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)

print(f"Best Hyperparameters: {best_params}")
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{classification_report}")

y_pred_prob = best_logistic_regression.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

validation_df = pd.read_csv(test_path)
validation_df.shape

validation_df.isnull().sum()



# replacing the null values with empty string
validation_df = validation_df.fillna('')



validation_df['content'] = validation_df['author']+' '+ validation_df['title']



validation_df['content'] = validation_df['content'].apply(lambda content: apply_stemming(content, stemmer))



#separating the features (stemmed contnent) and label
X_validation = validation_df['content'].values

# convert text data to its numerical representation
vectorizer = TfidfVectorizer(max_features=9000)
vectorizer.fit(X_validation)

X_validation = vectorizer.transform(X_validation)

X_validation.shape

# Make predictions on the test set
y_validation_pred = best_logistic_regression.predict(X_validation)

y_validation_pred.shape

# Evaluate the model
submit_df = pd.read_csv(submit_path)
y_validation_test = submit_df['label']
accuracy = accuracy_score(y_true=y_validation_test, y_pred=y_validation_pred)

print(f"Best Hyperparameters: {best_params}")
print(f"Accuracy: {accuracy}")

final_sub = pd.DataFrame()
final_sub['id']=submit_df['id']
final_sub['label'] = y_validation_pred
final_sub.to_csv('submission.csv',index=False)

wordcloud = WordCloud(width=800, height=400).generate(' '.join(train_df['content']))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

