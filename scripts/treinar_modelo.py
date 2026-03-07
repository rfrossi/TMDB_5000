"""
Script de treinamento do modelo Random Forest para predição de ROI de filmes.
Utiliza exclusivamente variáveis conhecidas antes da obra existir (features pré-lançamento):
  - budget        : orçamento planejado
  - runtime       : duração prevista (minutos)
  - release_month : mês de estreia
  - genre_encoded : gênero principal codificado

O target (ROI) é calculado como (revenue - budget) / budget * 100 (%).
Filtro de budget > 0 e revenue > 0 aplicado apenas para viabilizar o cálculo do ROI.
Outliers de budget/runtime identificados no Card 07 são mantidos na análise.

Uso:
    poetry run python scripts/treinar_modelo.py

Pré-requisito:
    O dataset 'tmdb_5000_pronto.csv' deve ter sido gerado pelo script data_preparation.py na pasta data/.
"""

import ast
import os
import sys
import warnings

# Garantir saída UTF-8 no Windows (evita UnicodeEncodeError no cp1252)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import joblib
import matplotlib
matplotlib.use("Agg")  # Backend sem display — evita erro de Tcl/Tk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
# 1. Carregar dados
# ---------------------------------------------------------------------------
if not os.path.exists(DATA_PATH):
    print(f"ERRO: Arquivo não encontrado: {DATA_PATH}")
    print("Execute o script de preparação primeiro para gerar data/tmdb_5000_pronto.csv:")
    print("  poetry run python scripts/data_preparation.py")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
print(f"[1/8] Dataset carregado: {df.shape[0]} filmes, {df.shape[1]} colunas")

# ---------------------------------------------------------------------------
# 2. Feature Engineering — variáveis pré-lançamento
# ---------------------------------------------------------------------------
print("[2/8] Feature Engineering...")

# Sazonalidade: extrair mês de lançamento
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_month'] = df['release_date'].dt.month

# Extrair gênero principal (primeiro da lista)
def extract_main_genre(genres_str):
    try:
        if pd.isna(genres_str):
            return 'Unknown'
        genres = ast.literal_eval(genres_str)
        if isinstance(genres, list) and len(genres) > 0:
            return genres[0]
        return 'Unknown'
    except Exception:
        return 'Unknown'

df['main_genre'] = df['genres'].apply(extract_main_genre)

# Codificação ordinal do gênero (para o modelo e para persistência)
genres_sorted = sorted(df['main_genre'].unique())
genre_to_code = {genre: code for code, genre in enumerate(genres_sorted)}
df['genre_encoded'] = df['main_genre'].map(genre_to_code)

print(f"      Gêneros únicos identificados: {len(genres_sorted)}")

# ---------------------------------------------------------------------------
# 3. Criar coluna alvo ROI — filtro de budget > 0 e revenue > 0
# ---------------------------------------------------------------------------
# Filtro aplicado APENAS para viabilizar o cálculo do ROI (budget=0 impossibilita divisão)
# Os limites de outliers de budget/runtime do Card 07 são preservados (não são excluídos)
df_roi = df[(df['budget'] > 0) & (df['revenue'] > 0)].copy()
df_roi['roi'] = (df_roi['revenue'] - df_roi['budget']) / df_roi['budget'] * 100

print(f"      Registros com budget > 0 e revenue > 0: {len(df_roi)} (de {len(df)} totais)")

FEATURES = ["budget", "runtime", "release_month", "genre_encoded"]
TARGET = "roi"

# Remover NaNs nas features e no target
df_model = df_roi[FEATURES + [TARGET]].dropna()
print(f"      Registros apos remocao de NaNs nas features: {len(df_model)}")
print(f"      Features: {FEATURES}")
roi_p1  = df_model[TARGET].quantile(0.01)
roi_p99 = df_model[TARGET].quantile(0.99)
print(f"      ROI bruto — Mediana: {df_model[TARGET].median():.1f}% | p1: {roi_p1:.1f}% | p99: {roi_p99:.1f}%")

# Transformação log1p para estabilizar o target (ROI >= -100%, logo roi+100 >= 0)
# log1p(roi + 100) comprime os extremos sem remover nenhum registro.
# A inversão no momento da predicao e: expm1(y_log) - 100
df_model = df_model.copy()
df_model['roi_log'] = np.log1p(np.clip(df_model[TARGET] + 100, 0.001, None))

X = df_model[FEATURES].values
y = df_model['roi_log'].values   # treinar no espaço log

# ---------------------------------------------------------------------------
# 4. Divisão treino/teste
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"[3/8] Divisão 80/20 — Treino: {len(X_train)}, Teste: {len(X_test)}")

# ---------------------------------------------------------------------------
# 5. Configuração do Pipeline (StandardScaler → PCA → RandomForestRegressor)
# ---------------------------------------------------------------------------
# Determinar número de componentes PCA que retêm ≥ 95% da variância
scaler_temp = StandardScaler()
X_train_scaled = scaler_temp.fit_transform(X_train)
pca_temp = PCA().fit(X_train_scaled)
cumvar = np.cumsum(pca_temp.explained_variance_ratio_)
N_COMPONENTS = max(1, int(np.argmax(cumvar >= 0.95)) + 1)
# Com 4 features, PCA pode reter tudo em < 4 componentes; usar ao menos 1
N_COMPONENTS = min(N_COMPONENTS, len(FEATURES))

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=N_COMPONENTS, random_state=42)),
    ('rf', RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ))
])

print("[4/8] Treinando Pipeline (StandardScaler -> PCA -> RandomForestRegressor)...")
pipeline.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 6. Detalhes das transformações
# ---------------------------------------------------------------------------
print("[5/8] Detalhes do Processamento:")
print("      - StandardScaler aplicado internamente.")
pca_final = pipeline.named_steps['pca']
print("      - PCA Aplicado:")
print(f"        Componentes selecionados: {N_COMPONENTS} (≥95% variância)")
print(f"        Variância retida: {pca_final.explained_variance_ratio_.sum()*100:.2f}%")
for i, (var, acum) in enumerate(zip(
    pca_final.explained_variance_ratio_,
    np.cumsum(pca_final.explained_variance_ratio_)
)):
    print(f"        PC{i+1}: {var*100:.2f}%  (acumulado: {acum*100:.2f}%)")
print("      - RandomForestRegressor treinado (200 arvores, profundidade=10).")

# ---------------------------------------------------------------------------
# 7. Avaliação — métricas no espaço ROI original (%)
# ---------------------------------------------------------------------------
y_pred_log = pipeline.predict(X_test)

# Inverter a transformação: expm1(y_log) - 100 => ROI em %
y_pred_roi = np.expm1(y_pred_log) - 100
y_test_roi = np.expm1(y_test) - 100

mae  = mean_absolute_error(y_test_roi, y_pred_roi)
rmse = np.sqrt(mean_squared_error(y_test_roi, y_pred_roi))
r2   = r2_score(y_test, y_pred_log)  # R² no espaço log (mais interpretavel)

cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2", n_jobs=-1)

print("\n[6/8] Avaliacao do modelo (Dados de Teste — Regressao de ROI):")
print(f"      MAE (ROI%)   : {mae:.1f}%")
print(f"      RMSE (ROI%)  : {rmse:.1f}%")
print(f"      R2 (espaco log): {r2:.4f}")
print(f"      CV R2 (5-fold): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# ---------------------------------------------------------------------------
# 8. Salvar modelo
# ---------------------------------------------------------------------------
os.makedirs(MODELS_DIR, exist_ok=True)

modelo_exportacao = {
    "pipeline": pipeline,
    "features": FEATURES,
    "n_components": N_COMPONENTS,
    "genre_to_code": genre_to_code,
    "genres_list": genres_sorted,
    "target_transform": "log1p(roi+100)",   # documentar inversão necessária
    "metricas": {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "cv_r2_media": cv_scores.mean(),
        "cv_r2_std": cv_scores.std(),
    },
}

joblib.dump(modelo_exportacao, MODEL_PATH)
tamanho_kb = os.path.getsize(MODEL_PATH) / 1024
print(f"[7/8] Modelo salvo em: {MODEL_PATH} ({tamanho_kb:.1f} KB)")

# ---------------------------------------------------------------------------
# 9. Gráficos
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Previsto vs Real (espaço log, para visualização limpa)
axes[0].scatter(y_test, y_pred_log, alpha=0.4, color='steelblue', edgecolors='none', s=20)
lims = [min(y_test.min(), y_pred_log.min()), max(y_test.max(), y_pred_log.max())]
axes[0].plot(lims, lims, 'r--', linewidth=1.5, label='Ideal')
axes[0].set_xlabel("log1p(ROI+100) Real")
axes[0].set_ylabel("log1p(ROI+100) Previsto")
axes[0].set_title("Previsto vs Real — ROI (escala log)", fontweight="bold")
axes[0].legend()

# Importância dos componentes PCA
rf_final = pipeline.named_steps['rf']
importancias = rf_final.feature_importances_
labels_pc = [f"PC{i+1}" for i in range(N_COMPONENTS)]
sorted_idx = np.argsort(importancias)[::-1]

axes[1].bar(
    [labels_pc[i] for i in sorted_idx],
    importancias[sorted_idx],
    color="steelblue",
    edgecolor="black",
)
axes[1].set_title("Importância dos Componentes PCA", fontweight="bold")
axes[1].set_ylabel("Importância")

plt.tight_layout()
grafico_path = os.path.join(MODELS_DIR, "avaliacao_modelo_roi.png")
plt.savefig(grafico_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[8/8] Gráfico salvo em: {grafico_path}")

print("\nConcluído com sucesso!")
