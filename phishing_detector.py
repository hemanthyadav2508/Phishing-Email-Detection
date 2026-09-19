import pandas as pd
import re
import numpy as np
from scipy.sparse import hstack

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. Feature Extraction Function
# ============================================================

def extract_email_features(text):
    text = str(text)

    # Count URLs
    url_count = len(
        re.findall(r'https?://\S+|www\.\S+', text)
    )

    # Suspicious keywords
    suspicious_keywords = [
        "urgent",
        "verify",
        "verification",
        "password",
        "account",
        "suspended",
        "click",
        "login",
        "bank",
        "confirm",
        "security",
        "winner",
        "prize",
        "claim",
        "payment",
        "update",
        "free",
        "limited",
        "expire"
    ]

    keyword_count = sum(
        text.lower().count(keyword)
        for keyword in suspicious_keywords
    )

    # Count email addresses
    email_count = len(
        re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    )

    # Count digits
    digit_count = sum(
        char.isdigit() for char in text
    )

    # Email length
    email_length = len(text)

    return [
        url_count,
        keyword_count,
        email_count,
        digit_count,
        email_length
    ]


# ============================================================
# 2. Load the Main Dataset
# ============================================================

data = pd.read_csv("Phishing_validation_emails (2).csv")

print("Dataset loaded successfully!")
print("Total emails:", len(data))


# ============================================================
# 3. Prepare Text and Labels
# ============================================================

X = data["Email Text"]

y = data["Email Type"].replace({
    "Phishing Email": "Phishing",
    "Safe Email": "Safe"
})


# ============================================================
# 4. Extract Email-Based Features
# ============================================================

email_features = X.apply(
    extract_email_features
)

email_features = np.array(
    email_features.tolist()
)

print("\nEmail feature extraction completed!")
print(
    "Features: URL count, suspicious keywords, "
    "email addresses, digits, email length"
)


# ============================================================
# 5. Split Dataset into Training and Testing
# ============================================================

X_train, X_test, y_train, y_test, features_train, features_test = train_test_split(
    X,
    y,
    email_features,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining emails:", len(X_train))
print("Testing emails:", len(X_test))


# ============================================================
# 6. TF-IDF Feature Extraction
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF feature extraction completed!")


# ============================================================
# 7. Combine Text + Email Features
# ============================================================

X_train_combined = np.hstack(
    (
        X_train_tfidf.toarray(),
        features_train
    )
)

X_test_combined = np.hstack(
    (
        X_test_tfidf.toarray(),
        features_test
    )
)

print("Text and email-based features combined!")


# ============================================================
# 8. Create Machine Learning Model
# ============================================================

model = LogisticRegression(
    max_iter=1000
)


# ============================================================
# 9. Train the Model
# ============================================================

model.fit(
    X_train_combined,
    y_train
)

print("Model training completed!")


# ============================================================
# 10. Test the Model
# ============================================================

y_pred = model.predict(
    X_test_combined
)


# ============================================================
# 11. Accuracy
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    "\nModel Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# ============================================================
# 12. Classification Report
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=["Phishing", "Safe"],
        zero_division=0
    )
)


# ============================================================
# 13. Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Phishing", "Safe"]
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 14. Plot Confusion Matrix
# ============================================================

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Phishing", "Safe"],
    yticklabels=["Phishing", "Safe"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Phishing Email Detection - Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png"
)

plt.show()

print("Confusion matrix saved successfully!")


# ============================================================
# 15. Test a Sample Email
# ============================================================

sample_email = """
URGENT! Your bank account has been suspended.
Click https://fake-bank-login.com immediately
to verify your password and account.
"""


# Extract text features
sample_tfidf = vectorizer.transform(
    [sample_email]
)

sample_email_features = np.array(
    [extract_email_features(sample_email)]
)

# Combine features
sample_combined = np.hstack(
    (
        sample_tfidf.toarray(),
        sample_email_features
    )
)

# Predict
sample_prediction = model.predict(
    sample_combined
)

print("\nSample Email Prediction:")
print(sample_prediction[0])


# ============================================================
# 16. Final Project Message
# ============================================================

print("\n========================================")
print("Phishing Email Detection Model Completed")
print("========================================")
print("Features Used:")
print("- TF-IDF textual features")
print("- URL count")
print("- Suspicious keyword count")
print("- Email address count")
print("- Digit count")
print("- Email length")
print("========================================")
# ==========================================
# 17. Test a New Email
# ==========================================

print("\n======================================")
print("   PHISHING EMAIL DETECTOR")
print("======================================")

new_email = input("\nEnter an email to check: ")

# Extract text features
new_text_tfidf = vectorizer.transform([new_email])

# Extract additional features
url_count = len(re.findall(r'https?://\S+|www\.\S+', new_email))

suspicious_words = [
    "urgent",
    "verify",
    "password",
    "account",
    "click",
    "login",
    "winner",
    "prize",
    "suspended",
    "confirm"
]

suspicious_count = sum(
    new_email.lower().count(word)
    for word in suspicious_words
)

email_count = len(
    re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', new_email)
)

digit_count = sum(char.isdigit() for char in new_email)

email_length = len(new_email)

# Combine features
new_features = np.array([[
    url_count,
    suspicious_count,
    email_count,
    digit_count,
    email_length
]])

new_combined = np.hstack(
    (
        new_text_tfidf.toarray(),
        new_features
    )
)
# Predict
prediction = model.predict(new_combined)

if prediction[0] == "Phishing":
    print("\n⚠️ Prediction: PHISHING EMAIL")
else:
    print("\n✅ Prediction: SAFE EMAIL")