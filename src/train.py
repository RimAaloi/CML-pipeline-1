from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
import joblib
import os

# Créer les dossiers nécessaires
os.makedirs('outputs', exist_ok=True)
os.makedirs('metrics', exist_ok=True)

print("=== CHARGEMENT DU DATASET IRIS ===")
iris = load_iris()
X, y = iris.data, iris.target

# Split des données
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("=== ENTRAÎNEMENT DU MODÈLE ===")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prédictions et métriques
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
loss = log_loss(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"Log Loss: {loss:.4f}")

# Sauvegarde du modèle et des métriques
joblib.dump(model, 'outputs/iris_model.pkl')

with open('metrics/accuracy.txt', 'w') as f:
    f.write(f"{accuracy:.4f}")

with open('metrics/loss.txt', 'w') as f:
    f.write(f"{loss:.4f}")

print("✅ Modèle sauvegardé: outputs/iris_model.pkl")