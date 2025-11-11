# Master’s Qualification Work 2025
## "Intelligent Phishing Identification System"

---

## Project Description
To save space, the folder containing pkl files has been archived in a ZIP file.
To run the system, extract the folder, open a terminal in VS Code, and execute:

```bash
python app.py
```

After that, the system’s web interface will open in a browser at: http://127.0.0.1:5000/

The relevance of this topic lies in the fact that phishing attacks are among the most common types of cybercrime, aimed at obtaining confidential user information through social engineering and forged web resources. Each year, the number of such attacks increases, causing significant damage to both individual users and large organizations. Therefore, the development of intelligent systems for automatic detection of phishing messages and URLs is an important direction for improving cybersecurity and preventing cyber threats.

The intelligent phishing identification system is designed to enhance cybersecurity, timely warn users of potential threats, and reduce the risk of unauthorized access to confidential information. To achieve this, machine learning methods, text data processing, and URL analysis are employed.

The object of research is the processes of automatic detection and classification of phishing messages in digital communication systems, as well as fraudulent URLs used in phishing attacks.

The subject of research is machine learning algorithms (Gradient Boosting, Random Forest, Logistic Regression, Support Vector Machine, Naive Bayes) for identifying phishing messages and URLs based on analysis of textual and URL features.

The architecture of the developed intelligent phishing identification system is modular, ensuring flexibility, scalability, and ease of integration with other analytical services. The system includes a user interface implemented with HTML, CSS, and JavaScript; a backend on Flask (Python), which handles user requests and performs message and URL classification; modules for text and URL preprocessing (cleaning, language detection, translation, TF-IDF vectorization); ML blocks using NB, LR, SVM, RF, and GB algorithms; a recommendation and feedback module; and a data storage system for models and results.

![Схема структури системи](https://github.com/inaprel3/PhisIdent/raw/main/picture/SchemeStructure.png)

## How the Program Works

The intelligent phishing identification system is implemented in Python using the Flask framework, which handles HTTP requests, integrates ML models, and interacts with the frontend. The main file, app.py, manages server launch, request routing (/check_message, /check_url, /feedback), loads trained models for classifying text messages (LR, Multinomial NB, SVM) and URLs (RF, GB), processes user input (URL validation, language detection, translation to English for model compatibility, vectorization), generates classification results and recommendations, handles user feedback, and automatically opens the web interface in a browser. Additional modules (messages.py, url.py, train_kaggle_datasets.py, train_recommendation_model.py) handle data preparation, model training, and evaluation, integrating open and custom datasets to improve classification accuracy and generate recommendations for potential phishing messages and URLs. The frontend in HTML, CSS, and JavaScript provides an intuitive interface for data entry, displaying classification results, recommendations, and collecting feedback. All data, intermediate results, and trained models are stored in CSV and PKL files, allowing offline operation and reproducibility.

For testing and evaluating system performance, several open and custom datasets were used, containing phishing and legitimate messages and URLs. The main source is the “Phishing Dataset” from Hugging Face (https://huggingface.co/datasets/ealvaradob/phishing-dataset
), containing Mail (≈18,000 emails), SMS (≈5,971 messages), URL (≈800,000 addresses), and Website (≈80,000 pages) subsets, all pre-cleaned, balanced, and structured in two columns: “text” and “label”. A reduced “combined reduced” version is recommended for model training. To localize and improve system accuracy, custom files custom_messages.csv and custom_urls.csv were created, including English and Ukrainian messages, as well as legitimate and phishing URLs from real attack examples and popular services. Additionally, Kaggle datasets were used: “Phishing Email Dataset” (https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset?select=phishing_email.csv
) with over 82,000 emails for text model training, and “Phishing URL Classifier Dataset Cleaned” (https://www.kaggle.com/datasets/adityachaudhary1306/phishing-url-classifier-dataset-cleaned
) with ~11,000 URLs for structural feature-based classification. Combining these sources ensures representativeness, class balance, format and language diversity, improving accuracy, generalization, and reliability.

The frontend of the phishing detection system is a web interface using HTML, CSS, and JavaScript, enabling intuitive user interaction with ML models in real time. The main page includes two forms for checking text messages and URLs, with an input field, a “Check” button, and a result block showing the model output in JSON format. A color-coded indicator quickly conveys the threat level: green – safe, red – phishing, yellow – requires further verification. Recommendations for next steps are displayed below, and users can provide feedback to improve the recommendation component. Tooltips and input error messages enhance usability. A typical workflow: user inputs text or URL → clicks “Check” → backend classifies the object → results with risk level and recommendations are displayed, making the verification process clear, intuitive, and convenient even for non-experts.

System home page in browser:
![](https://github.com/inaprel3/PhisIdent/blob/main/picture/System%20home%20page%20in%20browser.png)

Safe messages and links:
![](https://github.com/inaprel3/PhisIdent/blob/main/picture/Secure%20messages%20and%20links.png)

Dangerous messages and links:
![](https://github.com/inaprel3/PhisIdent/blob/main/picture/Dangerous%20messages%20and%20links.png)

Empty input fields for messages and links:
![](https://github.com/inaprel3/PhisIdent/blob/main/picture/Empty%20fields%20for%20entering%20messages%20and%20links.png)

### ML Model Results

#### Message Classification (SMS + Email + Custom)

| Model                    | Class          | Precision | Recall | F1-score | Support |
|--------------------------|----------------|-----------|--------|----------|---------|
| **Naive Bayes (NB)**     | 0 (safe)       | 0.96      | 0.96   | 0.96     | 2496    |
|                          | 1 (phishing)   | 0.93      | 0.94   | 0.93     | 1539    |
|                          | **accuracy**   |           |        | 0.95     | 4035    |
| **Logistic Regression**  | 0              | 0.96      | 0.97   | 0.96     | 2496    |
|                          | 1              | 0.96      | 0.93   | 0.94     | 1539    |
|                          | **accuracy**   |           |        | 0.96     | 4035    |
| **Support Vector Machine (SVM)** | 0        | 0.97      | 0.98   | 0.97     | 2496    |
|                          | 1              | 0.96      | 0.94   | 0.95     | 1539    |
|                          | **accuracy**   |           |        | 0.96     | 4035    |

#### URL Classification (Main + Custom)

| Model                    | Class          | Precision | Recall | F1-score | Support |
|--------------------------|----------------|-----------|--------|----------|---------|
| **Random Forest (RF)**   | 0 (safe)       | 0.97      | 0.98   | 0.97     | 89053   |
|                          | 1 (phishing)   | 0.97      | 0.96   | 0.97     | 78093   |
|                          | **accuracy**   |           |        | 0.97     | 167146  |
| **Gradient Boosting (GB)** | 0            | 0.90      | 0.93   | 0.92     | 89053   |
|                          | 1              | 0.92      | 0.88   | 0.90     | 78093   |
|                          | **accuracy**   |           |        | 0.91     | 167146  |

#### Kaggle Dataset Classification

**Phishing Email Dataset (recommendation model):**

| Category                    | Precision | Recall | F1-score | Support |
|-----------------------------|-----------|--------|----------|---------|
| account_blocked             | 1.00      | 0.67   | 0.80     | 55      |
| default                     | 0.93      | 0.98   | 0.95     | 5069    |
| payment_alert               | 0.93      | 0.38   | 0.54     | 113     |
| request_click_link          | 0.98      | 0.92   | 0.95     | 2237    |
| request_personal_data       | 0.82      | 0.79   | 0.80     | 1105    |
| **accuracy**                |           |        | 0.93     | 8579    |

**Phishing URL Dataset (Kaggle):**

| Model                    | Class          | Precision | Recall | F1-score | Support |
|--------------------------|----------------|-----------|--------|----------|---------|
| Random Forest (RF)       | 0              | 0.98      | 0.96   | 0.97     | 980     |
|                          | 1              | 0.97      | 0.98   | 0.98     | 1231    |
|                          | **accuracy**   |           |        | 0.97     | 2211    |
| Gradient Boosting (GB)   | 0              | 0.95      | 0.94   | 0.95     | 980     |
|                          | 1              | 0.95      | 0.96   | 0.96     | 1231    |
|                          | **accuracy**   |           |        | 0.95     | 2211    |

#### Recommendation model (Custom + SMS + Email)

| Category                    | Precision | Recall | F1-score | Support |
|-----------------------------|-----------|--------|----------|---------|
| account_blocked             | 0.00      | 0.00   | 0.00     | 4       |
| default                     | 0.84      | 0.83   | 0.83     | 879     |
| payment_alert               | 1.00      | 0.10   | 0.17     | 21      |
| request_click_link          | 0.91      | 0.82   | 0.87     | 418     |
| request_personal_data       | 0.83      | 0.57   | 0.68     | 217     |
| safe                        | 0.93      | 0.99   | 0.96     | 2496    |
| **accuracy**                |           |        | 0.91     | 4035    |

The PhisIdent system also collects user feedback, storing information about the object type, content, classification result, and user comment, which allows confirming or correcting the model’s decisions and further improving classification accuracy.

У результаті виконаної роботи було створено та протестовано інтелектуальну систему PhisIdent для виявлення фішингових повідомлень і URL-адрес, яка поєднує методи машинного навчання, обробки природної мови та аналізу вебресурсів. Розробка включала модульну архітектуру з бекендом на Python, фронтендом на HTML, CSS та JS і сховищем моделей у форматах PKL та CSV, що забезпечує гнучкість, масштабованість і простоту інтеграції в реальні середовища. Проведене тестування показало високу ефективність системи, точність класифікації більшості моделей перевищує 90–97%, що підтверджує адекватність вибору алгоритмів NB, LR, SVM, RF та GB та репрезентативність використаних наборів даних. Особливу роль у підвищенні точності та адаптивності системи відіграє механізм збору користувацького фідбеку, який дозволяє уточнювати результати класифікації та поступово вдосконалювати моделі. Отже, PhisIdent є практичним інструментом для підвищення кібербезпеки організацій, навчальних закладів і приватних користувачів, а подальший розвиток може включати інтеграцію з базами фішингових доменів, аналіз зображень та автоматичне оновлення моделей, що забезпечить ще більшу ефективність у виявленні сучасних загроз.
