import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
df = pd.read_csv("dataset_bmi_500.csv")
df.head(10)

le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label'])

X = df[['gender', 'tinggi', 'berat', 'imt']]
y = df['label_enc']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)

svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
pred_svm = svm.predict(X_test)

nb = GaussianNB()
nb.fit(X_train, y_train)
pred_nb = nb.predict(X_test)

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Logistic Regression
ConfusionMatrixDisplay.from_predictions(
    y_test, pred_logreg,
    display_labels=le.classes_,
    cmap="Blues"
)
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# SVM
ConfusionMatrixDisplay.from_predictions(
    y_test, pred_svm,
    display_labels=le.classes_,
    cmap="Greens"
)
plt.title("Confusion Matrix - SVM")
plt.show()

# Naive Bayes
ConfusionMatrixDisplay.from_predictions(
    y_test, pred_nb,
    display_labels=le.classes_,
    cmap="Oranges"
)
plt.title("Confusion Matrix - Naive Bayes")
plt.show()

# =========================
# RANDOM FOREST CLASSIFIER
# =========================

from sklearn.ensemble import RandomForestClassifier

# inisialisasi model
rf_model = RandomForestClassifier(
    n_estimators=200,         # jumlah pohon
    max_depth=None,           # kedalaman pohon
    random_state=42,
    class_weight="balanced",  # untuk data tidak seimbang
    n_jobs=-1
)

# training model
rf_model.fit(X_train, y_train)

print("Random Forest training selesai")
