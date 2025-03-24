from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('intents_dataset.csv')
df_clean = df.dropna(subset=['text'], axis=0)

X = df_clean['text']
y = df_clean['intent']


pipeline_rf = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

param_grid_rf = {
    'tfidf__max_df': [0.8, 0.9, 1.0],
    'tfidf__min_df': [1, 2],
    # You might tune the number of trees and max depth as well:
    'clf__max_depth': [None, 10, 20],
    'clf__min_samples_split': [2, 5]
}

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
grid_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=skf, scoring='accuracy')
grid_rf.fit(X, y)

print("Random Forest")
print(f"Best parameters: {grid_rf.best_params_}")
print(f"Best score: {grid_rf.best_score_:.2f}")
