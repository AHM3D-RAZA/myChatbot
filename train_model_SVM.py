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

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf', SVC(kernel='linear'))
])

param_grid = {
    'tfidf__max_df': [0.8, 0.9, 1.0],
    'tfidf__min_df': [1, 2],
    'clf__C': [0.1, 1, 10]
}

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
grid = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy')

grid.fit(X, y)

best_model = grid.best_estimator_
scores = cross_val_score(best_model, X, y, cv=skf)


print(f"Best parameters: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.2f}")
print(f"Final cross-validated accuracy: {scores.mean():.2f} (±{scores.std():.2f})")

joblib.dump(grid.best_estimator_, 'intent_classifier.joblib')