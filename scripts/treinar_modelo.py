"""
Script de treinamento do modelo Random Forest para predição de popularidade de filmes.
Este script utiliza o pipeline finalizado no Card 4, processando dados financeiros, 
reduzindo a dimensionalidade por PCA e persistindo o modelo para predições interativas.

Uso:
    poetry run python scripts/treinar_modelo.py

Pré-requisito:
    O dataset 'tmdb_5000_pronto.csv' deve ter sido gerado pelo script data_preparation.py na pasta data/.
"""

import os
import sys
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")  # Backend sem display — evita erro de Tcl/Tk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "tmdb_5000_pronto.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "random_forest_model.pkl")

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
# 2. Seleção e limpeza das features
# ---------------------------------------------------------------------------
FEATURES = ["budget", "revenue", "runtime", "vote_average", "vote_count"]
TARGET = "popularity"

FEATURES = [c for c in FEATURES if c in df.columns]
assert TARGET in df.columns, f"Coluna '{TARGET}' não encontrada no dataset."

df_model = df[FEATURES + [TARGET]].copy()
df_model.dropna(inplace=True)

for col in ["budget", "revenue"]:
    if col in df_model.columns:
        df_model = df_model[df_model[col] > 0]

print(f"[2/8] Registros após limpeza: {len(df_model)}")

# Variável alvo binária
mediana = df_model[TARGET].median()
df_model["popular"] = (df_model[TARGET] > mediana).astype(int)

X = df_model[FEATURES].values
y = df_model["popular"].values

print(f"      Features: {FEATURES}")
print(f"      Mediana de popularidade: {mediana:.2f}")
print(f"      Distribuição: {dict(zip(*np.unique(y, return_counts=True)))}")

# ---------------------------------------------------------------------------
# 3. Normalização com StandardScaler
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"[3/8] StandardScaler aplicado — média≈0, DP≈1 por feature")

# ---------------------------------------------------------------------------
# 4. PCA
# ---------------------------------------------------------------------------
pca_full = PCA()
pca_full.fit(X_scaled)
variancia_acumulada = np.cumsum(pca_full.explained_variance_ratio_)

# Componentes para >= 95% de variância
N_COMPONENTS = int(np.argmax(variancia_acumulada >= 0.95)) + 1

pca = PCA(n_components=N_COMPONENTS, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"[4/8] PCA aplicado:")
print(f"      Features originais: {X_scaled.shape[1]}")
print(f"      Componentes selecionados: {N_COMPONENTS} (≥95% variância)")
print(f"      Variância retida: {pca.explained_variance_ratio_.sum()*100:.2f}%")

for i, (var, acum) in enumerate(
    zip(pca.explained_variance_ratio_, np.cumsum(pca.explained_variance_ratio_))
):
    print(f"      PC{i+1}: {var*100:.2f}%  (acumulado: {acum*100:.2f}%)")

# ---------------------------------------------------------------------------
# 5. Divisão treino/teste
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[5/8] Divisão 80/20 — Treino: {len(X_train)}, Teste: {len(X_test)}")

# ---------------------------------------------------------------------------
# 6. Treinamento do Random Forest
# ---------------------------------------------------------------------------
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
rf_model.fit(X_train, y_train)
print("[6/8] Random Forest treinado (200 árvores, profundidade=10)")

# ---------------------------------------------------------------------------
# 7. Avaliação
# ---------------------------------------------------------------------------
y_pred = rf_model.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)
precisao = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

cv_scores = cross_val_score(rf_model, X_pca, y, cv=5, scoring="f1", n_jobs=-1)

print("[7/8] Avaliação do modelo:")
print(f"      Acurácia  : {acuracia*100:.2f}%")
print(f"      Precisão  : {precisao*100:.2f}%")
print(f"      F1-Score  : {f1*100:.2f}%")
print(f"      Recall    : {recall*100:.2f}%")
print(f"      CV F1 (5-fold): {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
print()
print(classification_report(y_test, y_pred, target_names=["Baixa Pop.", "Alta Pop."]))

# ---------------------------------------------------------------------------
# 8. Salvar modelo
# ---------------------------------------------------------------------------
os.makedirs(MODELS_DIR, exist_ok=True)

pipeline = {
    "scaler": scaler,
    "pca": pca,
    "model": rf_model,
    "features": FEATURES,
    "n_components": N_COMPONENTS,
    "mediana_popularidade": mediana,
    "metricas": {
        "acuracia": acuracia,
        "precisao": precisao,
        "f1_score": f1,
        "recall": recall,
        "cv_f1_media": cv_scores.mean(),
        "cv_f1_std": cv_scores.std(),
    },
}

joblib.dump(pipeline, MODEL_PATH)
tamanho_kb = os.path.getsize(MODEL_PATH) / 1024
print(f"[8/8] Modelo salvo em: {MODEL_PATH} ({tamanho_kb:.1f} KB)")

# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Baixa Pop.", "Alta Pop."]).plot(
    ax=axes[0], cmap="Blues", colorbar=False
)
axes[0].set_title("Matriz de Confusão", fontweight="bold")

importancias = rf_model.feature_importances_
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
grafico_path = os.path.join(MODELS_DIR, "avaliacao_modelo.png")
plt.savefig(grafico_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"      Gráfico salvo em: {grafico_path}")

print("\nConcluído com sucesso!")
