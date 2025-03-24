import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

df = pd.read_csv('intents_dataset.csv')
df_clean = df.dropna(subset=['text'], axis=0)

X = df_clean['text']
y = df_clean['intent']

# Encode class labels to numeric values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Define pipeline with XGBoost
pipeline_xgb = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
])

# Hyperparameter tuning grid
param_grid_xgb = {
    'tfidf__max_df': [0.8, 0.9, 1.0],
    'tfidf__min_df': [1, 2],
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [3, 5, 7]
}

# Stratified K-Fold Cross Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
grid_xgb = GridSearchCV(pipeline_xgb, param_grid_xgb, cv=skf, scoring='accuracy')

# Train model
grid_xgb.fit(X, y_encoded)

# Save best model
joblib.dump(grid_xgb.best_estimator_, 'intent_classifier_xgb.joblib')

# Print results
print(f"Best parameters: {grid_xgb.best_params_}")
print(f"Best score: {grid_xgb.best_score_:.2f}")

# Decode labels back to original class names
y_pred = grid_xgb.best_estimator_.predict(X)
y_pred_labels = label_encoder.inverse_transform(y_pred)

# Save Label Encoder for future use
joblib.dump(label_encoder, 'label_encoder.joblib')
