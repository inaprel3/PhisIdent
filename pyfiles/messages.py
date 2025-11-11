# pyfiles/messages.py
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import os

print("🔷 Завантаження основного датасету повідомлень (SMS + Email)...")
dataset = load_dataset("ealvaradob/phishing-dataset", "texts", trust_remote_code=True)
df_main = dataset['train'].to_pandas()

print("Перші 5 рядків основного датасету:")
print(df_main.head())

print("\n🔷 Завантаження кастомного датасету з csv/custom_messages.csv...")
custom_csv_path = os.path.join(os.path.dirname(__file__), '..', 'csv', 'custom_messages.csv')
df_custom = pd.read_csv(custom_csv_path)

print("Перші 5 рядків кастомного датасету:")
print(df_custom.head())

df = pd.concat([df_main, df_custom], ignore_index=True)
print(f"\nЗагальна кількість записів після об’єднання: {len(df)}")

df['label'] = df['label'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# Naive Bayes
nb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000)),
    ('clf', MultinomialNB())
])
nb_pipeline.fit(X_train, y_train)
y_pred_nb = nb_pipeline.predict(X_test)
print("\n--- Naive Bayes ---")
print(classification_report(y_test, y_pred_nb))

# Logistic Regression
lr_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000)),
    ('clf', LogisticRegression(max_iter=1000))
])
lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)
print("\n--- Logistic Regression ---")
print(classification_report(y_test, y_pred_lr))

# Support Vector Machine
svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000)),
    ('clf', SVC(probability=True))
])
svm_pipeline.fit(X_train, y_train)
y_pred_svm = svm_pipeline.predict(X_test)
print("\n--- Support Vector Machine ---")
print(classification_report(y_test, y_pred_svm))

joblib.dump(nb_pipeline, 'pkl/message_nb_model.pkl')
print("✅ Naive Bayes модель збережено як 'message_nb_model.pkl'")
joblib.dump(lr_pipeline, 'pkl/message_lr_model.pkl')
print("✅ Logistic Regression модель збережено як 'message_lr_model.pkl'")
joblib.dump(svm_pipeline, 'pkl/message_svm_model.pkl')
print("✅ Support Vector Machine модель збережено як 'message_svm_model.pkl'")
