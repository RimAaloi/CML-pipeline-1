from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
import joblib
import os

print("=== DÉMARRAGE ENTRAÎNEMENT ===")

# Créer dossiers
os.makedirs('outputs', exist_ok=True)
os.makedirs('metrics', exist_ok=True)

# Charger données
iris = load_iris()
X, y = iris.data, iris.target

# Diviser données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entraîner modèle
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Prédire
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Calculer métriques
accuracy = accuracy_score(y_test, y_pred)
loss = log_loss(y_test, y_pred_proba)

print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ Log Loss: {loss:.4f}")

# Sauvegarder
joblib.dump(model, 'outputs/iris_model.pkl')

with open('metrics/accuracy.txt', 'w') as f:
    f.write(f"{accuracy:.4f}")

with open('metrics/loss.txt', 'w') as f:
    f.write(f"{loss:.4f}")

print("✅ Modèle entraîné et sauvegardé!")