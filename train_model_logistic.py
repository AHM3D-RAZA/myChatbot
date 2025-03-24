from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import joblib

df = pd.read_csv('intents_dataset.csv')
df_clean = df.dropna(subset=['text'], axis=0)

X = df_clean['text']
y = df_clean['intent']

pipeline_lr = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=1000))
])

param_grid_lr = {
    'tfidf__max_df': [0.8, 0.9, 1.0],
    'tfidf__min_df': [1, 2],
    'clf__C': [0.1, 1, 10]
}

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
grid_lr = GridSearchCV(pipeline_lr, param_grid_lr, cv=skf, scoring='accuracy')
grid_lr.fit(X, y)

print("Logistic Regression")
print(f"Best parameters: {grid_lr.best_params_}")
print(f"Best score: {grid_lr.best_score_:.2f}")
