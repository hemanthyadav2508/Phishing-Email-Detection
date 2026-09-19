import pandas as pd
import re
import numpy as np

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

    # Suspicious keywords commonly found in phishing emails
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
        "update"
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
# 2. Load Dataset
# ============================================================

data = pd.read_csv("emails.csv")

print("Dataset loaded successfully!")
print("Total emails:", len(data))


# ============================================================
# 3. Separate Email Text and Labels
# ============================================================

X = data["text"]
y = data["label"]


# ============================================================
# 4. Extract URL and Email Features
# ============================================================

email_features = X.apply(extract_email_features)

email_features = np.array(
    email_features.tolist()
)

print("\nEmail feature extraction completed!")
print("Features: URL count, suspicious keywords, email addresses, digits, email length")


# ============================================================
# 5. Split Dataset
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
# 6. TF-IDF Text Feature Extraction
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
# 7. Combine TF-IDF + Email Features
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
# 9. Train Model
# ============================================================

model.fit(
    X_train_combined,
    y_train
)

print("Model training completed!")


# ============================================================
# 10. Predict Test Emails
# ============================================================

y_pred = model.predict(
    X_test_combined
)


# ============================================================
# 11. Calculate Accuracy
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
# 14. Confusion Matrix Graph
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

plt.savefig("confusion_matrix.png")

plt.show()

print("Confusion matrix saved successfully!")


# ============================================================
# 15. Validate Model on Unseen Emails
# ============================================================

validation_data = pd.read_csv(
    "Phishing_validation_emails (2).csv"
)

X_validation = validation_data["Email Text"]

# Convert validation labels to match training labels
y_validation = validation_data["Email Type"].replace(
    {
        "Phishing Email": "Phishing",
        "Safe Email": "Safe"
    }
)


# ============================================================
# 16. Extract Validation Email Features
# ============================================================

validation_features = X_validation.apply(
    extract_email_features
)

validation_features = np.array(
    validation_features.tolist()
)


# ============================================================
# 17. Convert Validation Text Using Existing TF-IDF
# ============================================================

X_validation_tfidf = vectorizer.transform(
    X_validation
)


# ============================================================
# 18. Combine Validation Features
# ============================================================

X_validation_combined = np.hstack(
    (
        X_validation_tfidf.toarray(),
        validation_features
    )
)


# ============================================================
# 19. Predict Validation Emails
# ============================================================

validation_pred = model.predict(
    X_validation_combined
)


# ============================================================
# 20. Validation Accuracy
# ============================================================

validation_accuracy = accuracy_score(
    y_validation,
    validation_pred
)

print(
    "\nValidation Accuracy:",
    round(validation_accuracy * 100, 2),
    "%"
)


# ============================================================
# 21. Validation Classification Report
# ============================================================

print("\nValidation Classification Report:")

print(
    classification_report(
        y_validation,
        validation_pred,
        zero_division=0
    )
)


# ============================================================
# 22. Validation Confusion Matrix
# ============================================================

validation_cm = confusion_matrix(
    y_validation,
    validation_pred,
    labels=["Phishing", "Safe"]
)

print("\nValidation Confusion Matrix:")
print(validation_cm)


# ============================================================
# 23. Save Validation Confusion Matrix
# ============================================================

plt.figure(figsize=(6, 5))

sns.heatmap(
    validation_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Phishing", "Safe"],
    yticklabels=["Phishing", "Safe"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Validation Confusion Matrix")

plt.tight_layout()

plt.savefig("validation_confusion_matrix.png")

plt.show()

print("Validation confusion matrix saved successfully!")


# ============================================================
# 24. Final Message
# ============================================================

print("\n========================================")
print("Phishing Email Detection Model Completed")
print("========================================")