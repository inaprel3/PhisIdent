# pyfiles/train_kaggle_datasets.py
import os
import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import kagglehub

print("🔷 Починаємо завантаження і тренування Kaggle датасетів")

# 1) Download Phishing Email Dataset
print("1) Завантаження Kaggle Phishing Email Dataset...")
email_path = kagglehub.dataset_download("naserabdullahalam/phishing-email-dataset")
print("Файли завантажено у:", email_path)

# Використовуємо phishing_email.csv
email_csv = os.path.join(email_path, "phishing_email.csv")
if not os.path.exists(email_csv):
    raise FileNotFoundError(f"Не знайдено файл phishing_email.csv у {email_path}")

df_email = pd.read_csv(email_csv)
df_email = df_email.rename(columns={'text_combined': 'text'})[['text', 'label']]
print("✅ Завантажено email дані, записів:", len(df_email))


# 2) Download Phishing URL Dataset
print("\n2) Завантаження Kaggle Phishing URL Dataset...")
url_path = kagglehub.dataset_download("adityachaudhary1306/phishing-url-classifier-dataset-cleaned")
print("Файли завантажено у:", url_path)

# Використовуємо dataset.csv
url_csv = os.path.join(url_path, "dataset.csv")
if not os.path.exists(url_csv):
    raise FileNotFoundError(f"Не знайдено файл dataset.csv у {url_path}")

df_url = pd.read_csv(url_csv)
print("✅ Завантажено URL дані, записів:", len(df_url))

# Заміна -1 → 0 у колонці Result
df_url['Result'] = df_url['Result'].replace({-1: 0}).astype(int)
X_url = df_url.drop(columns=['Result'])
y_url = df_url['Result']


# 3) Training recommendation model на email
print("\n3) Тренування recommendation model на email...")

# Беремо тільки фішингові листи
df_phish = df_email[df_email['label'] == 1].copy()

def assign_category(text):
    t = str(text).lower()
    if re.search(r"(click|link|посилання|перейдіть|натисніть)", t):
        return "request_click_link"
    if re.search(r"(name|address|credit card|card number|номер картки|персональні дані)", t):
        return "request_personal_data"
    if re.search(r"(blocked|suspended|блоковано|заблоковано)", t):
        return "account_blocked"
    if re.search(r"(payment|invoice|оплата|рахунок|billing)", t):
        return "payment_alert"
    return "default"

df_phish['category'] = df_phish['text'].apply(assign_category)

X = df_phish['text']
y = df_phish['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000))
])
pipeline.fit(X_train, y_train)

print("\n📊 Classification report (email):")
print(classification_report(y_test, pipeline.predict(X_test)))

os.makedirs('pkl', exist_ok=True)
joblib.dump(pipeline, 'pkl/recommendation_model_kaggle.pkl')
print("✅ recommendation_model_kaggle.pkl збережено")


# 4) Training URL models
print("\n4) Тренування URL моделей...")

X_train, X_test, y_train, y_test = train_test_split(X_url, y_url, test_size=0.2, random_state=42, stratify=y_url)

# RandomForest
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print("\n📊 RandomForest report:")
print(classification_report(y_test, rf.predict(X_test)))
joblib.dump(rf, 'pkl/url_rf_model_kaggle.pkl')
print("✅ url_rf_model_kaggle.pkl збережено")

# GradientBoosting
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
print("\n📊 GradientBoosting report:")
print(classification_report(y_test, gb.predict(X_test)))
joblib.dump(gb, 'pkl/url_gb_model_kaggle.pkl')
print("✅ url_gb_model_kaggle.pkl збережено")

print("\n🎯 Готово! Усі моделі успішно натреновані та збережені.")