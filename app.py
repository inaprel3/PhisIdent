# app.py
from flask import Flask, render_template, request, jsonify
import webbrowser
import threading
import joblib
import re
from langdetect import detect
from googletrans import Translator
from datetime import datetime
import os

app = Flask(__name__)

# Завантаження моделей
def safe_load(path, name):
    try:
        mdl = joblib.load(path)
        print(f"✅ Завантажено {name} з '{path}'")
        return mdl
    except Exception as e:
        print(f"⚠️ Не вдалося завантажити {name} з '{path}': {e}")
        return None

message_model = safe_load('pkl/message_lr_model.pkl', 'message_model (LR)')
recommendation_model = safe_load('pkl/recommendation_model.pkl', 'recommendation_model (local)')
recommendation_model_kaggle = safe_load('pkl/recommendation_model_kaggle.pkl', 'recommendation_model_kaggle')
url_model = safe_load('pkl/url_rf_model.pkl', 'url_model (RF)')
url_vectorizer = safe_load('pkl/url_vectorizer.pkl', 'url_vectorizer (TF-IDF for url)')
url_model_kaggle = safe_load('pkl/url_rf_model_kaggle.pkl', 'url_model_kaggle (RF)')
url_gb_model_kaggle = safe_load('pkl/url_gb_model_kaggle.pkl', 'url_gb_model_kaggle (GB)')

translator = Translator()

# Допоміжні функції
def is_valid_url(url: str) -> bool:
    pattern = re.compile(r'^(https?://)([^\s]+\.)+[^\s]{2,}', re.IGNORECASE)
    return pattern.match(url) is not None

def contains_only_url(text: str) -> bool:
    text = text.strip()
    if re.fullmatch(r'(https?://)?([^\s]+\.)+[^\s]{2,}(\/\S*)?', text, re.IGNORECASE):
        return True
    return False

DEFAULT_RECOMMENDATIONS = {
    "request_click_link": "⚠️ Не переходьте за посиланням! Це може бути шахрайство.",
    "request_personal_data": "⚠️ Не надавайте свої особисті або банківські дані.",
    "account_blocked": "⚠️ Ваш акаунт можуть використовувати шахраї. Не повідомляйте дані та перевірте офіційний сайт.",
    "payment_alert": "⚠️ Не здійснюйте оплату без перевірки достовірності повідомлення.",
    "default": "⚠️ Небезпечно довіряти цьому повідомленню!"
}

# Роути
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/check_message", methods=["POST"])
def check_message():
    data = request.json or {}
    message = (data.get("message") or "").strip()
    if message == "":
        return jsonify({"result": "⚠️ Введіть текст повідомлення для перевірки", "valid_message": False})
    if contains_only_url(message):
        return jsonify({"result": "❌ У полі повідомлення має бути текст, а не лише посилання.", "valid_message": False, "recommendation": ""})

    try:
        lang = detect(message)
    except Exception:
        lang = None

    if lang and lang != 'en':
        try:
            translation = translator.translate(message, src=lang, dest='en')
            message_en = translation.text
        except Exception as e:
            print(f"[WARN] Translation failed: {e}")
            message_en = message
    else:
        message_en = message

    if message_model is None:
        return jsonify({"result": "❌ Модель для класифікації повідомлень не завантажена.", "valid_message": False})

    try:
        pred = int(message_model.predict([message_en])[0])
    except Exception as e:
        print(f"[ERROR] message_model prediction failed: {e}")
        return jsonify({"result": "❌ Помилка під час передбачення", "valid_message": False})

    if pred == 0:
        return jsonify({"result": "Повідомлення безпечне", "valid_message": True, "recommendation": "✅ Можна довіряти повідомленню.", "category": None})

    cat = None
    used_rec_model = None
    if recommendation_model_kaggle:
        try:
            cat = recommendation_model_kaggle.predict([message_en])[0]
            used_rec_model = "recommendation_model_kaggle"
        except Exception as e:
            print(f"[WARN] recommendation_model_kaggle failed: {e}")
            cat = None
    if cat is None and recommendation_model:
        try:
            cat = recommendation_model.predict([message_en])[0]
            used_rec_model = "recommendation_model_local"
        except Exception as e:
            print(f"[WARN] recommendation_model (local) failed: {e}")
            cat = None
    if cat is None:
        cat = "default"

    recommendation = DEFAULT_RECOMMENDATIONS.get(cat, DEFAULT_RECOMMENDATIONS["default"])

    return jsonify({"result": "Фішингове повідомлення", "valid_message": True, "recommendation": recommendation, "category": cat, "used_recommendation_model": used_rec_model})

@app.route("/check_url", methods=["POST"])
def check_url():
    data = request.json or {}
    url = (data.get("url") or "").strip()
    if url == "":
        return jsonify({"result": "⚠️ Введіть URL-посилання для перевірки", "valid_url": False})
    if not is_valid_url(url):
        return jsonify({"result": "❌ Некоректний формат посилання", "valid_url": False})

    if url_vectorizer is None or url_model is None:
        return jsonify({"result": "❌ Модель для перевірки URL не готова на сервері.", "valid_url": False})

    try:
        url_vect = url_vectorizer.transform([url])
        pred = int(url_model.predict(url_vect)[0])
    except Exception as e:
        print(f"[ERROR] URL prediction failed: {e}")
        return jsonify({"result": "❌ Помилка під час аналізу посилання", "valid_url": False})

    if pred == 1:
        return jsonify({"result": "Фішингове посилання", "valid_url": True, "recommendation": "⚠️ Небезпечно переходити за цим посиланням!"})
    else:
        return jsonify({"result": "Посилання безпечне", "valid_url": True, "recommendation": "✅ Можна переходити за посиланням."})

# Роут: фідбек
@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json or {}
    feedback_type = data.get("type", "unknown")
    content = data.get("content", "")
    system_result = data.get("system_result", "")
    user_opinion = data.get("user_opinion", "")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n[FEEDBACK] {timestamp}")
    print(f"Тип: {feedback_type}")
    print(f"Контент: {content}")
    print(f"Результат системи: {system_result}")
    print(f"Відгук користувача: {user_opinion}\n")

    return jsonify({"message": "✅ Дякуємо за відгук! Його збережено."})

# Авто відкриття браузера
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == "__main__":
    threading.Timer(1.25, open_browser).start()
    app.run(debug=True)