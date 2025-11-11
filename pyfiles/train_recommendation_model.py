# pyfiles/train_recommendation_model.py
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
import joblib
from datasets import load_dataset
import os

print("🔷 Завантаження основного датасету повідомлень (SMS + Email)...")
dataset = load_dataset("ealvaradob/phishing-dataset", "texts", trust_remote_code=True)
df_main = dataset['train'].to_pandas()

print("🔷 Завантаження кастомного датасету з csv/custom_messages.csv...")
custom_csv_path = os.path.join('csv', 'custom_messages.csv')
df_custom = pd.read_csv(custom_csv_path)

df = pd.concat([df_main, df_custom], ignore_index=True)
df['label'] = df['label'].astype(int)
print(f"Загальна кількість записів після об’єднання: {len(df)}")

def assign_category(text, label):
    text = str(text).lower()

    # Безпечні повідомлення
    if label == 0:
        return "safe"

    # Фішингові категорії
    if re.search(r"(click|link|посилання|перейдіть|натисніть|confirm)", text):
        return "request_click_link"
    elif re.search(r"(name|address|credit card|card number|номер картки|персональні дані)", text):
        return "request_personal_data"
    elif re.search(r"(blocked|suspended|блоковано|заблоковано|тимчасово заблоковано|увійдіть)", text):
        return "account_blocked"
    elif re.search(r"(card|credit|карт|рахунок|платіж|оплат|invoice|billing)", text):
        return "payment_alert"
    elif re.search(r"(номер карт|банківськ|особист|персональн|введіть дані|введіть номер)", text):
        return "request_personal_data"
    else:
        return "default"

df['category'] = df.apply(lambda row: assign_category(row['text'], row['label']), axis=1)

print("\n📊 Розподіл категорій повідомлень:")
print(df['category'].value_counts())

ANTI_TRIGGER_WORDS = ["click", "link", "посилання", "натисніть", "confirm", "blocked", "payment", "invoice", "card", "credit"]

def apply_anti_trigger(row):
    text = str(row['text']).lower()
    cat = row['category']
    if cat != "safe":
        if not any(word in text for word in ANTI_TRIGGER_WORDS):
            return "default"  # зменшили вагу фішинговості
    return cat

df['category'] = df.apply(apply_anti_trigger, axis=1)

X = df['text']
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=7000)),
    ('clf', OneVsRestClassifier(LogisticRegression(max_iter=1500)))
])

print("\n🚀 Навчання моделі...")
pipeline.fit(X_train, y_train)

print("\n📈 Classification report:")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))

os.makedirs('pkl', exist_ok=True)
joblib.dump(pipeline, 'pkl/recommendation_model.pkl')
print("\n✅ Модель рекомендацій збережено як 'pkl/recommendation_model.pkl'")
