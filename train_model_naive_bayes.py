from sklearn.naive_bayes import MultinomialNB
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

pipeline_nb = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf', MultinomialNB())
])

param_grid_nb = {
    'tfidf__max_df': [0.8, 0.9, 1.0],
    'tfidf__min_df': [1, 2],
    # For Naive Bayes, you can experiment with smoothing parameter alpha
    'clf__alpha': [0.5, 1.0, 1.5]
}

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
grid_nb = GridSearchCV(pipeline_nb, param_grid_nb, cv=skf, scoring='accuracy')
grid_nb.fit(X, y)

print("Multinomial Naive Bayes")
print(f"Best parameters: {grid_nb.best_params_}")
print(f"Best score: {grid_nb.best_score_:.2f}")
