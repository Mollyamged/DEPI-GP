import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib


# -------- Load Data --------
df = pd.read_csv(r"C:\Users\Marley Amged\Downloads\archive\heart.csv")   # أو اسم الملف عندك

# Encode categorical columns
cat_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features & Target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------- Train Model --------
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "heart_model.pkl")

# Save encoders
joblib.dump(label_encoders, "encoders.pkl")


# -------- Evaluate --------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))

# -------- Prediction Function --------
def predict_heart(data_dict):
    df_input = pd.DataFrame([data_dict])
    # Encode input using trained label encoders
    for col in cat_cols:
        df_input[col] = label_encoders[col].transform(df_input[col])
    pred = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][1]
    return pred, prob


