# Hindi Movie Sentiment Analysis

An end-to-end web application for **Hindi movie review sentiment analysis** using Machine Learning. The project includes a Flask backend, React/Vite frontend, Hindi movie-review dataset, preprocessing pipeline, sentiment classification, movie recommendations, and model evaluation.

---

## Project Overview

The system analyzes Hindi movie reviews and classifies them into three sentiment categories:

- **Positive**
- **Negative**
- **Neutral**

The project also provides movie-wise sentiment statistics and a watchability/recommendation score through a Hindi user interface.

### Main Technologies

- **Python**
- **Flask**
- **scikit-learn**
- **Pandas**
- **NumPy**
- **React**
- **Vite**
- **Tailwind CSS**
- **TF-IDF**
- **Logistic Regression**
- **Machine Learning evaluation metrics**

---

## Project Structure

```text
movie-sentiment-analysis/
│
├── backend/
│   ├── app.py                 # Flask API and application entry point
│   ├── model.py               # Model training/loading and prediction
│   ├── preprocessing.py       # Text preprocessing and dataset handling
│   ├── recommendation.py      # Watchability score and recommendations
│   ├── requirements.txt       # Python dependencies
│   └── models/
│       └── .gitkeep           # Trained models are generated here
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # Main React application
│   │   ├── main.jsx           # React entry point
│   │   ├── index.css          # Global styles
│   │   └── services/
│   │       └── api.js         # Backend API communication
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
│
├── dataset/
│   └── movie_reviews.csv      # Hindi movie review dataset
│
└── README.md
```

---

# 1. Dataset

The project uses a Hindi movie-review dataset stored at:

```text
dataset/movie_reviews.csv
```

The dataset contains movie information and review-related fields. The application automatically preprocesses the review text before passing it to the machine-learning model.

### Important Note

The supplied dataset does **not contain manually annotated sentiment labels**. Therefore, sentiment labels are generated using a **weak-supervision approach** based on Hindi sentiment keywords and emojis.

Examples of positive indicators include words/emojis associated with:

- अच्छा
- शानदार
- बेहतरीन
- जबरदस्त
- पसंद
- ❤️
- 😍
- 👍

Examples of negative indicators include:

- बकवास
- बेकार
- घटिया
- बुरा
- नफरत
- 😡
- 😢
- 👎

These automatically generated labels are then used as the target classes for supervised learning.

> **Note:** Since the labels are automatically generated rather than human annotated, evaluation results should be interpreted as performance against the generated labels, not as a benchmark against human ground truth.

---

# 2. Machine Learning Pipeline

The overall workflow is:

```text
Hindi Movie Reviews
        │
        ▼
Text Preprocessing
        │
        ▼
Sentiment Keyword + Emoji Labelling
        │
        ▼
Train/Test Split
        │
        ▼
TF-IDF Feature Extraction
        │
        ▼
Supervised ML Algorithms
        │
        ├── Logistic Regression
        ├── Linear SVM
        ├── Multinomial Naive Bayes
        └── Random Forest
        │
        ▼
Predictions
        │
        ▼
Evaluation
        │
        ├── Accuracy
        ├── Precision
        ├── Recall
        ├── F1 Score
        ├── Confusion Matrix
        └── ROC Curve / AUC
```

---

# 3. Supervised Learning Algorithms

The project can compare multiple supervised learning algorithms using the same TF-IDF features.

## Logistic Regression

Logistic Regression is used as a linear classification model for predicting the sentiment class from TF-IDF features.

## Linear SVM

Linear Support Vector Machine finds a decision boundary that separates the sentiment classes in the TF-IDF feature space.

## Multinomial Naive Bayes

Multinomial Naive Bayes is commonly used for text classification and works with word-frequency-based representations such as TF-IDF.

## Random Forest

Random Forest combines multiple decision trees to perform classification.

---

# 4. Evaluation Metrics

The algorithms are compared using:

### Precision

Precision measures how many of the reviews predicted as a particular class actually belong to that class.

```text
Precision = TP / (TP + FP)
```

### Recall

Recall measures how many of the actual reviews belonging to a class were correctly identified.

```text
Recall = TP / (TP + FN)
```

### F1 Score

F1-score is the harmonic mean of precision and recall.

```text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

For the multiclass experiment, **weighted averages** are used for the overall Precision, Recall and F1-score so that class distribution is taken into account.

---

# 5. ROC Curve and AUC

Because the sentiment problem has three classes:

```text
Negative
Neutral
Positive
```

the ROC analysis uses a **One-vs-Rest (OvR)** approach.

A separate ROC curve is considered for each sentiment class:

```text
Negative vs Rest
Neutral vs Rest
Positive vs Rest
```

The **Area Under the Curve (AUC)** is also calculated.

The ROC curve plots:

- True Positive Rate (TPR)
- False Positive Rate (FPR)

A random classifier is represented by the diagonal reference line.

The ROC/AUC comparison can be used alongside Precision, Recall and F1-score when analyzing the supervised learning models.

---

# 6. Model Comparison

The supervised learning experiment produces a comparison table similar to:

| Algorithm | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | Calculated | Calculated | Calculated | Calculated |
| Linear SVM | Calculated | Calculated | Calculated | Calculated |
| Multinomial Naive Bayes | Calculated | Calculated | Calculated | Calculated |
| Random Forest | Calculated | Calculated | Calculated | Calculated |

The values should be generated by running the experiment rather than manually entering them.

---

# 7. Backend Setup

## Requirements

Make sure the following are installed:

- Python 3.9+
- pip

Open a terminal in the project folder.

### Windows

```bash
cd backend

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

python app.py
```

### Linux / macOS

```bash
cd backend

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

python app.py
```

The Flask backend runs on:

```text
http://localhost:5000
```

---

# 8. Frontend Setup

Open another terminal:

```bash
cd frontend

npm install

npm run dev
```

The Vite development server normally runs at:

```text
http://localhost:5173
```

Open that address in your browser.

The backend should be started **before** using the frontend.

---

# 9. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/dashboard` | Returns dashboard statistics and model metrics |
| GET | `/api/movies` | Returns movie-wise sentiment information |
| GET | `/api/movie/<name>` | Returns details and reviews for a movie |
| POST | `/api/predict` | Predicts sentiment for a new review |
| POST | `/api/retrain` | Retrains the sentiment model |

### Example Prediction Request

```json
{
  "review": "यह फिल्म बहुत शानदार है"
}
```

The API returns the predicted sentiment and confidence information.

---

# 10. Retraining the Model

If the dataset is changed, the model can be retrained using:

```bash
curl -X POST http://localhost:5000/api/retrain
```

The trained model is cached inside:

```text
backend/models/
```

This avoids retraining every time the application starts.

---

# 11. Using a Different Dataset

A different CSV dataset can be used by replacing:

```text
dataset/movie_reviews.csv
```

The preprocessing code supports common column names for movie and review information.

Possible movie-name columns:

```text
movie_name
movie
title
film
```

Possible review-text columns:

```text
raw_text
review
review_text
text
comment
```

Possible rating columns:

```text
rating
stars
score
```

---

# 12. Frontend Features

The web application provides:

- Hindi-language user interface
- Movie-wise sentiment analysis
- Positive/negative/neutral sentiment percentages
- Review prediction
- Model performance information
- Movie watchability/recommendation score
- Sample movie reviews
- Dashboard statistics
- Movie-level analysis

---

# 13. Watchability Score

The application calculates a sentiment-based watchability score.

The basic calculation is:

```text
raw_score =
(positive_count - negative_count) / total_reviews
```

The score is then converted to a 0–100 scale:

```text
watchability_score =
(raw_score + 1) / 2 × 100
```

This score represents the sentiment expressed in the available reviews. It should **not** be interpreted as an objective measurement of a movie's quality.

---

# 14. Limitations

1. The dataset is relatively small.
2. Sentiment labels are generated using weak supervision rather than human annotation.
3. The Hindi sentiment keyword and emoji lexicons are limited.
4. Hindi text can contain spelling variations, slang, code-mixing and transliterated words that are difficult to classify.
5. Model performance can change significantly with a larger or manually labelled dataset.
6. ROC-AUC values should be interpreted in the context of the multiclass One-vs-Rest setup.
7. The application is intended primarily as an academic/project demonstration rather than a production-grade sentiment analysis system.

---

# 15. Recommended Improvements

For a more reliable sentiment-analysis system, the following improvements can be considered:

- Use a larger Hindi review dataset.
- Create manually annotated sentiment labels.
- Expand the Hindi sentiment lexicon.
- Handle Hindi-English code-mixed reviews.
- Experiment with character and word n-grams.
- Compare additional classifiers.
- Use pretrained Hindi language models such as IndicBERT or MuRIL.
- Perform cross-validation instead of relying only on a single train/test split.
- Tune model hyperparameters.
- Evaluate the model on an independent human-labelled test set.

---

# 16. Academic Experiment

For the supervised-learning comparison experiment, the recommended workflow is:

```text
1. Load dataset
2. Preprocess Hindi review text
3. Generate sentiment labels
4. Split data into 80% training and 20% testing
5. Convert text to TF-IDF vectors
6. Train Logistic Regression
7. Train Linear SVM
8. Train Multinomial Naive Bayes
9. Train Random Forest
10. Predict test-set labels
11. Calculate Precision
12. Calculate Recall
13. Calculate F1-score
14. Generate confusion matrices
15. Generate multiclass ROC curves
16. Calculate ROC-AUC
17. Compare all algorithms
```

---

## Conclusion

This project demonstrates an end-to-end **Hindi Movie Sentiment Analysis** system using supervised machine learning. It combines text preprocessing, weakly supervised sentiment labelling, TF-IDF feature extraction, multiple classification algorithms, evaluation metrics, ROC-AUC analysis, and a React-based web interface.

The project can be extended by replacing the automatically generated labels with a larger **human-annotated Hindi sentiment dataset**, which would provide a more reliable basis for evaluating model performance.

---

# 17. User Accounts and Live Reviews

The application now supports persistent user accounts and user-generated movie reviews.

## Authentication

Users can:

- Create an account using a Hindi/Devanagari username.
- Use an English password containing letters, numbers and symbols.
- Log in and log out.
- Keep their login session while using the website.

Passwords are stored as secure hashes using Werkzeug; plain-text passwords are never stored.

Authentication data is stored in:

```text
backend/movie_sentiment.db
```

## Writing a Review

A logged-in user can open **समीक्षा लिखें** and enter:

1. Film name
2. Movie review

When the user submits the review, the backend:

```text
User Review
     ↓
Text Preprocessing
     ↓
TF-IDF + Logistic Regression
     ↓
Sentiment + Confidence
     ↓
Save Review to SQLite
     ↓
Recalculate Movie Statistics
     ↓
Return Updated Watchability Score
```

The review is immediately analysed and saved.

## Existing and New Movies

A user can review a movie already present in the original dataset.

Users can also enter a completely new movie name. If that movie does not already exist, the application automatically creates a movie record and attaches the submitted review to it.

The new movie then appears in the movie list and dashboard.

## Important Model Behaviour

User predictions are **not automatically used as new training labels**. This avoids a feedback loop where the model would train on its own predictions.

The original sentiment model continues to be trained from the project's existing dataset. User reviews are stored separately and are included when calculating movie-level sentiment statistics and watchability scores.

The original dataset remains unchanged:

```text
dataset/movie_reviews.csv
```

## New API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/auth/register` | Create a user account |
| POST | `/api/auth/login` | Log in |
| POST | `/api/auth/logout` | Log out |
| GET | `/api/auth/me` | Check current login session |
| POST | `/api/reviews` | Analyse and save a user review |
| GET | `/api/reviews/my` | Return the logged-in user's reviews |
| GET | `/api/health` | Backend health check |

## Windows Setup

### Backend

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

### Frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

Keep the Flask backend running on:

```text
http://localhost:5000
```

## Database Reset

If you want to start authentication and user reviews from scratch, stop Flask and delete:

```text
backend/movie_sentiment.db
```

The database and its tables will be recreated automatically the next time `python app.py` is started.
