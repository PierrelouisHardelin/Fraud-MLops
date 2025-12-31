# Fraud-MLops — Credit Card Fraud Detection (2023)

Ce projet met en place un pipeline **MLOps minimal** (data → train → API) pour détecter la fraude à la carte bancaire à partir d’un dataset anonymisé de transactions **2023** (≈ 550k+ lignes, features `V1..V28`, `Amount`, cible `Class`).

## 1) Structure du projet

```
Fraud-MLops/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── model.joblib
│   ├── metrics.json
│   └── reference_stats.json
├── src/
│   ├── data/
│   │   ├── clean_transform.py
│   │   └── make_dataset.py
│   ├── models/
│   │   ├── train.py
│   │   └── predict.py
│   └── api/
│       ├── main.py
│       └── schemas.py
├── requirements.txt
└── .gitignore
```

## 2) Prérequis

- Python 3.10+
- (optionnel) AWS CLI si vous lisez directement depuis S3

Installation des dépendances :

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3) Configuration AWS (si dataset sur S3)

Si le dataset est sur S3, configurez vos credentials :

```bash
aws configure
aws sts get-caller-identity
```

> Le script `make_dataset` accepte un chemin local **ou** un URI S3 au format `s3://bucket/key.csv`.

## 4) Génération du dataset (ETL / preprocessing)

Le script :

- lit un CSV (local ou S3),
- effectue un nettoyage léger + feature engineering (`log_amount`),
- fait un split **shuffle + stratifié** (train/test),
- écrit `train.parquet` et `test.parquet` dans `data/processed/`,
- calcule des stats de référence (`reference_stats.json`).

### 4.1 Depuis un fichier local

```bash
python -m src.data.make_dataset --input data/raw/creditcard_2023.csv --fmt parquet
```

### 4.2 Depuis S3

```bash
python -m src.data.make_dataset --input s3://YOUR_BUCKET/YOUR_KEY/creditcard_2023.csv --fmt parquet
```

Sorties attendues :

- `data/processed/train.parquet`
- `data/processed/test.parquet`
- `models/reference_stats.json`

## 5) Entraîner le modèle

Le training :

- charge `train.parquet` et `test.parquet`,
- entraîne un modèle sklearn (Logistic Regression + scaler),
- sauvegarde le modèle et les métriques.

```bash
python -m src.models.train --fmt parquet
```

Sorties attendues :

- `models/model.joblib`
- `models/metrics.json`
- `models/artifact.json`

## 6) Lancer l’API (FastAPI)

Démarrer l’API :

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints :

- `GET /health` : état de l’API + modèle chargé
- `GET /metrics` : métriques train/test (`models/metrics.json`)
- `GET /reference-stats` : stats de référence (`models/reference_stats.json`)
- `POST /predict` : prédiction sur une transaction

Interface Swagger :

- http://localhost:8000/docs

## 7) Tests rapides

### 7.1 Healthcheck

**PowerShell**

```powershell
Invoke-RestMethod http://localhost:8000/health
```

**curl**

```bash
curl http://localhost:8000/health
```

### 7.2 Voir les métriques

```bash
curl http://localhost:8000/metrics
```

### 7.3 Prédire une transaction (exemple minimal)

L’API attend un JSON :

- `request_id` (optionnel)
- `features` : dictionnaire contenant au minimum `Amount` (et idéalement `V1..V28`)

Exemple (l’API calcule `log_amount` si non fourni) :

```bash
curl -X POST "http://localhost:8000/predict"   -H "Content-Type: application/json"   -d '{"request_id":"demo-1","features":{"Amount":123.45}}'
```

### 7.4 Prédire avec une vraie ligne du test set

Générer un payload réaliste depuis `test.parquet` :

```bash
python -c "import pandas as pd, json; df=pd.read_parquet('data/processed/test.parquet'); x=df.drop(columns=['Class']).iloc[0].to_dict(); print(json.dumps({'request_id':'row0','features':x}))"
```

Copier-coller le JSON obtenu dans `/docs` → `/predict` → **Try it out**, ou l’envoyer via curl.

## 8) Notes / Hypothèses

- Le dataset 2023 utilisé est anonymisé (features `V1..V28`) et contient la cible `Class`.
- Le split est **shuffle + stratifié** pour garantir une distribution cohérente des classes dans train/test.
- Le seuil de décision est par défaut `0.5` (modifiable dans `src/models/train.py` et stocké avec le modèle).

## 9) Commandes “full run” (pipeline complet)

```bash
# 1) Build dataset
python -m src.data.make_dataset --input s3://YOUR_BUCKET/YOUR_KEY/creditcard_2023.csv --fmt parquet

# 2) Train
python -m src.models.train --fmt parquet

# 3) Run API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
