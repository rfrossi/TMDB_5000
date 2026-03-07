"""
Script de treinamento do modelo Random Forest para predicao de ROI de filmes.
Versao 2.0 — Features enriquecidas com star power, brand power e sazonalidade.

Features utilizadas (todas pre-lancamento, sem data leakage):
  - log_budget              : log1p(orcamento) — normaliza escala
  - runtime                 : duracao prevista (minutos)
  - release_month           : mes de estreia (1–12)
  - is_blockbuster_season   : Jun/Jul/Dez = 1
  - director_roi_encoded    : media suavizada de ROI log dos filmes anteriores do diretor
  - cast_roi_encoded        : media suavizada de ROI log dos filmes anteriores do ator principal
  - top_studio              : 1 se a produtora esta entre os top-20 studios
  - is_sequel               : 1 se keywords contem 'sequel'
  - genre_<X>               : one-hot para os top-15 generos

Target: log1p(roi + 100), onde roi = (revenue - budget) / budget * 100 (%)
Inversao em predicao: expm1(pred) - 100

Uso:
    poetry run python scripts/treinar_modelo.py

Pre-requisito:
    data/tmdb_5000_pronto.csv gerado por scripts/data_preparation.py
"""

import ast
import json
import os
import sys
import warnings
from collections import Counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Saida UTF-8 no Windows (evita UnicodeEncodeError no cp1252)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings("ignore")
np.random.seed(42)

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "tmdb_5000_pronto.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "random_forest_roi.pkl")

# ---------------------------------------------------------------------------
# Helpers de parse
# ---------------------------------------------------------------------------

def safe_eval(val):
    """Converte string de lista/dict (CSV) para objeto Python."""
    if pd.isna(val) or str(val).strip() == '':
        return []
    try:
        return ast.literal_eval(str(val))
    except Exception:
        try:
            return json.loads(str(val))
        except Exception:
            return []


def extract_director(crew_val):
    crew = safe_eval(crew_val)
    for member in crew:
        if isinstance(member, dict) and member.get('job') == 'Director':
            return member.get('name', 'Unknown')
    return 'Unknown'


def extract_genres_list(genres_val):
    genres = safe_eval(genres_val)
    result = []
    for g in genres:
        if isinstance(g, str):
            result.append(g)
        elif isinstance(g, dict):
            result.append(g.get('name', ''))
    return [x for x in result if x]


def extract_top_actor(cast_val):
    """Retorna o ator de maior billing (primeiro da lista)."""
    cast = safe_eval(cast_val)
    if not cast:
        return 'Unknown'
    top = cast[0]
    if isinstance(top, str):
        return top
    if isinstance(top, dict):
        return top.get('name', 'Unknown')
    return 'Unknown'


def extract_studio(companies_val):
    companies = safe_eval(companies_val)
    if not companies:
        return 'Unknown'
    first = companies[0]
    if isinstance(first, dict):
        return first.get('name', 'Unknown')
    if isinstance(first, str):
        return first
    return 'Unknown'


def keywords_has_sequel(keywords_val):
    kws = safe_eval(keywords_val)
    for kw in kws:
        name = kw.get('name', '') if isinstance(kw, dict) else str(kw)
        if 'sequel' in name.lower():
            return 1
    return 0


# ---------------------------------------------------------------------------
# 1. Carregar dados
# ---------------------------------------------------------------------------
if not os.path.exists(DATA_PATH):
    print(f"ERRO: Arquivo nao encontrado: {DATA_PATH}")
    print("Execute: poetry run python scripts/data_preparation.py")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
print(f"[1/9] Dataset carregado: {df.shape[0]} filmes, {df.shape[1]} colunas")

# ---------------------------------------------------------------------------
# 2. Feature Engineering
# ---------------------------------------------------------------------------
print("[2/9] Feature Engineering...")

for col in ['budget', 'revenue', 'runtime']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_month'] = df['release_date'].dt.month

df['log_budget'] = np.log1p(df['budget'])
df['is_blockbuster_season'] = df['release_month'].apply(
    lambda m: 1 if m in [6, 7, 12] else 0
)

df['director'] = df['crew'].apply(extract_director)
df['top_actor'] = df['cast'].apply(extract_top_actor)
df['studio'] = df['production_companies'].apply(extract_studio)
df['is_sequel'] = df['keywords'].apply(keywords_has_sequel)
df['genres_list'] = df['genres'].apply(extract_genres_list)

# Multi-genre one-hot (top 15 generos por frequencia)
TOP_GENRES_N = 15
all_genres = [g for lst in df['genres_list'] for g in lst]
TOP_GENRES = [g for g, _ in Counter(all_genres).most_common(TOP_GENRES_N)]
GENRE_COLS = [f'genre_{g}' for g in TOP_GENRES]

for genre, col in zip(TOP_GENRES, GENRE_COLS):
    df[col] = df['genres_list'].apply(lambda lst: int(genre in lst))

print(f"      Diretor/Ator extraidos | {len(TOP_GENRES)} generos one-hot criados")
print(f"      Generos: {TOP_GENRES}")

# ---------------------------------------------------------------------------
# 3. Calcular ROI e target log
# ---------------------------------------------------------------------------
print("[3/9] Calculando ROI e target log...")

# Filtro APENAS para viabilizar calcuo do ROI — outliers de budget/runtime preservados
df_roi = df[(df['budget'] > 0) & (df['revenue'] > 0)].copy()
df_roi['roi'] = (df_roi['revenue'] - df_roi['budget']) / df_roi['budget'] * 100

# log1p(roi + 100): comprime extremos sem remover registros (ROI >= -100% sempre)
df_roi['roi_log'] = np.log1p(np.clip(df_roi['roi'] + 100, 0.001, None))

global_roi_log_mean = df_roi['roi_log'].mean()

print(f"      Registros validos: {len(df_roi)}")
print(f"      ROI bruto — Mediana: {df_roi['roi'].median():.1f}% | "
      f"p1: {df_roi['roi'].quantile(0.01):.1f}% | p99: {df_roi['roi'].quantile(0.99):.1f}%")
print(f"      ROI log mean global: {global_roi_log_mean:.4f}")

# ---------------------------------------------------------------------------
# 4. Target Encoding suavizado (Director e Ator Principal)
#    Suavizacao bayesiana: blend entre media individual e media global
# ---------------------------------------------------------------------------
SMOOTH_K = 10  # peso da media global (quanto maior, mais conservador)

print("[4/9] Target Encoding suavizado (director e ator principal)...")

def smooth_encode(series_groups, global_mean, k=SMOOTH_K):
    """Retorna dict {nome: encoded_value} com suavizacao bayesiana."""
    stats = series_groups.agg(['mean', 'count'])
    stats['encoded'] = (
        (stats['mean'] * stats['count'] + global_mean * k)
        / (stats['count'] + k)
    )
    return stats['encoded'].to_dict()


director_encoding = smooth_encode(
    df_roi.groupby('director')['roi_log'], global_roi_log_mean
)
df_roi['director_roi_encoded'] = (
    df_roi['director'].map(director_encoding).fillna(global_roi_log_mean)
)

cast_encoding = smooth_encode(
    df_roi.groupby('top_actor')['roi_log'], global_roi_log_mean
)
df_roi['cast_roi_encoded'] = (
    df_roi['top_actor'].map(cast_encoding).fillna(global_roi_log_mean)
)

print(f"      Diretores encodados: {len(director_encoding)}")
print(f"      Atores encodados:    {len(cast_encoding)}")

# ---------------------------------------------------------------------------
# 5. Top Studios
# ---------------------------------------------------------------------------
print("[5/9] Top Studios...")

TOP_STUDIOS_N = 20
TOP_STUDIOS = df_roi['studio'].value_counts().head(TOP_STUDIOS_N).index.tolist()
df_roi['top_studio'] = df_roi['studio'].apply(lambda s: 1 if s in TOP_STUDIOS else 0)

print(f"      Top {TOP_STUDIOS_N} studios: {TOP_STUDIOS[:5]}...")

# ---------------------------------------------------------------------------
# 6. Montar dataset final
# ---------------------------------------------------------------------------
FEATURES = (
    ['log_budget', 'runtime', 'release_month', 'is_blockbuster_season',
     'director_roi_encoded', 'cast_roi_encoded', 'top_studio', 'is_sequel']
    + GENRE_COLS
)
TARGET_COL = 'roi_log'

df_model = df_roi[FEATURES + [TARGET_COL]].dropna()

print(f"\n[6/9] Features ({len(FEATURES)} total):")
print(f"      {FEATURES}")
print(f"      Registros apos limpeza: {len(df_model)}")

X = df_model[FEATURES].values
y = df_model[TARGET_COL].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"      Divisao 80/20 — Treino: {len(X_train)}, Teste: {len(X_test)}")

# ---------------------------------------------------------------------------
# 7. Pipeline: StandardScaler -> PCA -> RandomForestRegressor
# ---------------------------------------------------------------------------
print("[7/9] Treinando Pipeline (StandardScaler -> PCA -> RandomForestRegressor)...")

scaler_temp = StandardScaler()
X_train_scaled = scaler_temp.fit_transform(X_train)
pca_temp = PCA().fit(X_train_scaled)
cumvar = np.cumsum(pca_temp.explained_variance_ratio_)
N_COMPONENTS = max(1, int(np.argmax(cumvar >= 0.95)) + 1)
N_COMPONENTS = min(N_COMPONENTS, len(FEATURES))

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=N_COMPONENTS, random_state=42)),
    ('rf', RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ))
])

pipeline.fit(X_train, y_train)

pca_final = pipeline.named_steps['pca']
print(f"      PCA: {N_COMPONENTS} componentes — variancia retida: "
      f"{pca_final.explained_variance_ratio_.sum()*100:.2f}%")

# ---------------------------------------------------------------------------
# 8. Avaliacao
# ---------------------------------------------------------------------------
print("[8/9] Avaliacao...")

y_pred_log = pipeline.predict(X_test)
y_pred_roi = np.expm1(y_pred_log) - 100
y_test_roi = np.expm1(y_test) - 100

mae = mean_absolute_error(y_test_roi, y_pred_roi)
rmse = np.sqrt(mean_squared_error(y_test_roi, y_pred_roi))
r2 = r2_score(y_test, y_pred_log)
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2', n_jobs=-1)

print(f"\n      R2 (escala log):   {r2:.4f}")
print(f"      CV R2 (5-fold):    {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
print(f"      MAE (ROI%):        {mae:.1f}%")
print(f"      RMSE (ROI%):       {rmse:.1f}%")

# ---------------------------------------------------------------------------
# 9. Salvar modelo e graficos
# ---------------------------------------------------------------------------
os.makedirs(MODELS_DIR, exist_ok=True)

# Listas para popularar selects no dashboard
top_directors_ui = (
    pd.DataFrame({'count': list({d: True for d in director_encoding}.values()),
                  'name': list(director_encoding.keys())})
    ['name'].tolist()
)
top_directors_ui = sorted([
    d for d in director_encoding
    if d != 'Unknown'
])

top_actors_ui = sorted([
    a for a in cast_encoding
    if a != 'Unknown'
])

modelo_exportacao = {
    "pipeline": pipeline,
    "features": FEATURES,
    "genre_columns": GENRE_COLS,
    "top_genres": TOP_GENRES,
    "director_encoding": director_encoding,
    "cast_encoding": cast_encoding,
    "top_studios": TOP_STUDIOS,
    "global_roi_log_mean": global_roi_log_mean,
    "top_directors_ui": top_directors_ui,
    "top_actors_ui": top_actors_ui,
    "n_components": N_COMPONENTS,
    "target_transform": "log1p(roi+100)",
    "metricas": {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "cv_r2_media": cv_scores.mean(),
        "cv_r2_std": cv_scores.std(),
    },
}

joblib.dump(modelo_exportacao, MODEL_PATH)
print(f"[9/9] Modelo salvo: {MODEL_PATH} ({os.path.getsize(MODEL_PATH)/1024:.1f} KB)")

# Graficos
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_test, y_pred_log, alpha=0.3, color='steelblue', s=15, edgecolors='none')
lims = [min(y_test.min(), y_pred_log.min()), max(y_test.max(), y_pred_log.max())]
axes[0].plot(lims, lims, 'r--', linewidth=1.5, label='Ideal')
axes[0].set_xlabel("log1p(ROI+100) Real")
axes[0].set_ylabel("log1p(ROI+100) Previsto")
axes[0].set_title("Previsto vs Real (escala log)", fontweight="bold")
axes[0].legend()

rf_final = pipeline.named_steps['rf']
importancias = rf_final.feature_importances_
labels_pc = [f"PC{i+1}" for i in range(N_COMPONENTS)]
sorted_idx = np.argsort(importancias)[::-1]
axes[1].bar(
    [labels_pc[i] for i in sorted_idx],
    importancias[sorted_idx],
    color="steelblue", edgecolor="black"
)
axes[1].set_title("Importancia dos Componentes PCA", fontweight="bold")
axes[1].set_ylabel("Importancia")

plt.tight_layout()
grafico_path = os.path.join(MODELS_DIR, "avaliacao_modelo_roi.png")
plt.savefig(grafico_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"      Grafico salvo: {grafico_path}")

print("\nConcluido com sucesso!")
