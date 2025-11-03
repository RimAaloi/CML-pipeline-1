import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
import pandas as pd
import joblib
import numpy as np

# Charger le modèle et les données
model = joblib.load('outputs/iris_model.pkl')
iris = load_iris()
X, y = iris.data, iris.target

# Re-split pour cohérence
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

y_pred = model.predict(X_test)

# 1. Matrice de confusion
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)
plt.title('Matrice de Confusion - Iris Dataset')
plt.ylabel('Vraie Classe')
plt.xlabel('Classe Prédite')
plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Importance des features
plt.figure(figsize=(10, 6))
feature_importance = model.feature_importances_
features = iris.feature_names
indices = np.argsort(feature_importance)[::-1]

plt.barh(range(len(features)), feature_importance[indices])
plt.yticks(range(len(features)), [features[i] for i in indices])
plt.title('Importance des Features - Random Forest')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Distribution des classes
plt.figure(figsize=(8, 6))
class_dist = pd.Series(y).value_counts()
plt.bar(iris.target_names, class_dist.values, 
        color=['skyblue', 'lightcoral', 'lightgreen'])
plt.title('Distribution des Classes - Iris Dataset')
plt.ylabel('Nombre d\'échantillons')
for i, v in enumerate(class_dist.values):
    plt.text(i, v + 1, str(v), ha='center', va='bottom')
plt.tight_layout()
plt.savefig('outputs/class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Graphiques générés dans le dossier outputs/")