# pyfiles/url.py
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
import joblib
import os

print("🔷 Завантаження основного датасету URL...")
url_dataset = load_dataset("ealvaradob/phishing-dataset", "urls", trust_remote_code=True)
df_main = url_dataset['train'].to_pandas()

print("Перші 5 рядків основного датасету:")
print(df_main.head())

print("\n🔷 Завантаження кастомного датасету з csv/custom_urls.csv...")
custom_csv_path = os.path.join(os.path.dirname(__file__), '..', 'csv', 'custom_urls.csv')
df_custom = pd.read_csv(custom_csv_path)

print("Перші 5 рядків кастомного датасету:")
print(df_custom.head())

df = pd.concat([df_main, df_custom], ignore_index=True)
print(f"\nЗагальна кількість записів після об’єднання: {len(df)}")

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=5000)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

rf_model = RandomForestClassifier(random_state=42)
gb_model = GradientBoostingClassifier(random_state=42)

print("\n--- Random Forest ---")
rf_model.fit(X_train_vect, y_train)
print(classification_report(y_test, rf_model.predict(X_test_vect)))

print("\n--- Gradient Boosting ---")
gb_model.fit(X_train_vect, y_train)
print(classification_report(y_test, gb_model.predict(X_test_vect)))

joblib.dump(rf_model, 'pkl/url_rf_model.pkl')
print("✅ Random Forest модель збережена як 'url_rf_model.pkl'")

joblib.dump(gb_model, 'pkl/url_gb_model.pkl')
print("✅ Gradient Boosting модель збережена як 'url_gb_model.pkl'")

joblib.dump(vectorizer, 'pkl/url_vectorizer.pkl')
print("✅ Векторизатор збережено як 'url_vectorizer.pkl'")
