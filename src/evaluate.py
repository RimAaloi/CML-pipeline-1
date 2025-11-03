import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
import joblib
import numpy as np

print("=== DÉMARRAGE ÉVALUATION ===")

# Charger modèle
model = joblib.load('outputs/iris_model.pkl')
iris = load_iris()

# Recréer données de test
from sklearn.model_selection import train_test_split
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

y_pred = model.predict(X_test)

# 1. Matrice de confusion
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de Confusion - Iris')
plt.savefig('outputs/confusion_matrix.png')
plt.close()

# 2. Importance des features
plt.figure(figsize=(10, 6))
importance = model.feature_importances_
plt.barh(iris.feature_names, importance)
plt.title('Importance des Features')
plt.savefig('outputs/feature_importance.png')
plt.close()

print("✅ Graphiques générés!")

# Créer rapport CML
with open('report.md', 'w') as f:
    f.write("# 🌸 Rapport CML - Classification Iris\n\n")
    f.write("## 📊 Résultats\n\n")
    
    with open('metrics/accuracy.txt', 'r') as acc_file:
        accuracy = acc_file.read().strip()
    
    with open('metrics/loss.txt', 'r') as loss_file:
        loss = loss_file.read().strip()
    
    f.write(f"- **Accuracy**: {accuracy}\n")
    f.write(f"- **Log Loss**: {loss}\n\n")
    f.write("## 📈 Visualisations\n\n")
    f.write("### Matrice de Confusion\n")
    f.write("![Matrice de Confusion](outputs/confusion_matrix.png)\n\n")
    f.write("### Importance des Features\n")
    f.write("![Importance des Features](outputs/feature_importance.png)\n")

print("✅ Rapport CML créé!")