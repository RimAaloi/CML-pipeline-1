# test_model.py
import joblib
import numpy as np

# Charger le modèle entraîné
model = joblib.load('outputs/iris_model.pkl')

# Données de test (exemple: [sepal_length, sepal_width, petal_length, petal_width])
test_data = [
    [5.1, 3.5, 1.4, 0.2],  # Doit prédire Setosa
    [6.7, 3.0, 5.2, 2.3],  # Doit prédire Virginica
    [5.9, 3.0, 4.2, 1.5]   # Doit prédire Versicolor
]

predictions = model.predict(test_data)

# Noms des classes
class_names = ['Setosa', 'Versicolor', 'Virginica']

print("=== PRÉDICTIONS DU MODÈLE ===")
for i, pred in enumerate(predictions):
    print(f"Exemple {i+1}: {test_data[i]} → {class_names[pred]}")