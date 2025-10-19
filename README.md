Projet MLOps : Prédiction de risque de crédit

1. Description
Ce projet met en œuvre un pipeline MLOps complet pour la prédiction du risque de crédit à partir de données de prêts.

2. Structure du projet
MLOps/
│
├── mlruns/                 # Suivi des expériences MLflow
├── models/                 # Modèles enregistrés
│
├── Loan_Data.csv           # Jeu de données de base
├── projet_final.ipynb      # Notebook principal (EDA + entraînement)
├── main_app.py             # Script d'application principale
│
├── requirements.txt        # Dépendances du projet
├── .gitignore              # Fichiers à ignorer par Git
│
├── README.md               # Documentation du projet
├── LICENSE                 # Licence GPL-3.0
├── SECURITY.md             # Politique de sécurité
└── CODE_OF_CONDUCT.md      # Charte de contribution

3. Installation
A) Cloner le dépôt
git clone https://github.com/mimimonna/MLOps.git
cd MLOps

B) Créer et activer un environnement virtuel
python3 -m venv venv
source venv/bin/activate   # sous Mac/Linux
venv\Scripts\activate      # sous Windows

C) Installer les dépendances 
pip install -r requirements.txt


4. Exécution du projet
A) Notebook d'analyse et de modélisation
- Explorer et nettoyer les données (Loan_Data.csv)
- Entraîner un modèle de classification (ex. logistic regression, random forest…)
- Évaluer les performances et sauvegarder le modèle dans le dossier /models

B) Application principale
Le script main_app.py charge le modèle final et permet de faire des prédictions à partir de nouvelles données :
python main_app.py

**5. SUIVI D'EXPÉRIENCE (MLfLOW)**
Le dossier mlruns contient les logs générés par MLflow.
Chaque run conserve :
- les hyperparamètres utilisés
- les scores du modèle
- les artefacts (modèle entraîné, métriques, etc.)

Pour lancer l’interface MLflow :
mlflow ui
Puis ouvrir http://localhost:5000

**6. BONNES PRATIQUES**
Le projet suit les principes :
- Versionnement Git pour la traçabilité
- MLflow pour la reproductibilité des expériences
- Sécurité & éthique (voir SECURITY.md et CODE_OF_CONDUCT.md)
- Licence GPL-3.0 pour une utilisation ouverte et responsable

Projet réalisé par Monna DABO, Hugo BONNEMAISON et Liliane DIM, dans le cadre de la formation Sorbonne Data Analystics.
