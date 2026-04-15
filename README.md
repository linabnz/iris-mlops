# Mini projet MLOps — Breast Cancer Classifier

**Dataset :** Breast Cancer (`sklearn.datasets.load_breast_cancer`) — 569 échantillons, 30 features, 2 classes  
**Stack :** Python 3.11, scikit-learn, FastAPI, Docker, GitHub Actions, GHCR

## Structure

```
.
├── app/
│   ├── __init__.py
│   ├── train.py       # Entraînement + sauvegarde model.pkl
│   └── api.py         # FastAPI: /health + /predict
├── Dockerfile
├── requirements.txt
└── .github/workflows/ci.yml
```

## Dataset

Breast Cancer Wisconsin Dataset intégré dans scikit-learn. Aucun fichier à télécharger.

- **Échantillons :** 569
- **Features (30) :** mesures géométriques des cellules tumorales (rayon, texture, périmètre, aire...)
- **Classes (2) :** `malignant` (0), `benign` (1)

## Lancer en local

```bash
pip install -r requirements.txt
python -m app.train
uvicorn app.api:app --reload
```

API sur `http://localhost:8000` — doc interactive sur `/docs`.

## Lancer avec Docker

```bash
docker build -t cancer-api .
docker run -p 8000:8000 cancer-api
```

## Endpoints

### `GET /health`
```json
{"status": "ok", "model_loaded": true}
```

### `POST /predict`
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "mean_radius": 17.99, "mean_texture": 10.38, "mean_perimeter": 122.8,
    "mean_area": 1001.0, "mean_smoothness": 0.1184, "mean_compactness": 0.2776,
    "mean_concavity": 0.3001, "mean_concave_points": 0.1471, "mean_symmetry": 0.2419,
    "mean_fractal_dimension": 0.07871, "radius_error": 1.095, "texture_error": 0.9053,
    "perimeter_error": 8.589, "area_error": 153.4, "smoothness_error": 0.006399,
    "compactness_error": 0.04904, "concavity_error": 0.05373, "concave_points_error": 0.01587,
    "symmetry_error": 0.03003, "fractal_dimension_error": 0.006193, "worst_radius": 25.38,
    "worst_texture": 17.33, "worst_perimeter": 184.6, "worst_area": 2019.0,
    "worst_smoothness": 0.1622, "worst_compactness": 0.6656, "worst_concavity": 0.7119,
    "worst_concave_points": 0.2654, "worst_symmetry": 0.4601, "worst_fractal_dimension": 0.1189
  }'
```
Réponse :
```json
{"prediction": 0, "label": "malignant"}
```

## Pipeline CI

- **Push sur `feature/**`** → installe les dépendances + entraîne le modèle.
- **Push sur `develop` ou `main`** → entraîne + build l'image Docker + publie sur `ghcr.io/<owner>/cancer-api`.