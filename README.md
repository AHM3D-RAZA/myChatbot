# myChatbot
This chatbot is trained on an intents.json file which contains tags with patterns and responses.
I trained the bot with SVM, XGBoost, Random Forest and Logistic Regression. Their results are attached below. I also developed a simple GUI for the bot using HTML, CSS and used Flask for localhost server.

          -------RESULTS-------
Logistic Regression
Best parameters: {'clf__C': 10, 'tfidf__max_df': 0.8, 'tfidf__min_df': 2}
Best score: 0.82

Random Forest
Best parameters: {'clf__max_depth': None, 'clf__min_samples_split': 5, 'tfidf__max_df': 0.8, 'tfidf__min_df': 2}
Best score: 0.77


Multinomial Naive Bayes
Best parameters: {'clf__alpha': 1.0, 'tfidf__max_df': 0.8, 'tfidf__min_df': 1}
Best score: 0.79

XGBoost
Best parameters: {'clf__max_depth': 7, 'clf__n_estimators': 100, 'tfidf__max_df': 0.8, 'tfidf__min_df': 2}
Best score: 0.71

Best parameters: {'clf__C': 1, 'tfidf__max_df': 0.8, 'tfidf__min_df': 1}
Best score: 0.82
Final cross-validated accuracy: 0.81 (±0.06)
