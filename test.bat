@echo off
echo === ACTIVATION ENVIRONNEMENT VIRTUEL ===
call venv\Scripts\activate

echo === INSTALLATION DEPENDANCES ===
pip install -r requirements.txt

echo === ENTRAINEMENT MODELE ===
python src\train.py

echo === EVALUATION MODELE ===
python src\evaluate.py

echo === VERIFICATION FICHIERS ===
dir outputs\
dir metrics\

echo === AFFICHAGE METRIQUES ===
type metrics\accuracy.txt
type metrics\loss.txt

echo === TEST PREDICTIONS ===
python test_model.py

pause