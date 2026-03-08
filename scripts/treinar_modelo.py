"""
Script de treinamento do modelo Random Forest para predição de sucesso financeiro de filmes.
Classifica se um filme terá sucesso binário (revenue >= 2 * budget) usando features
numéricas e categóricas com ColumnTransformer, SMOTE e Random Forest.

Correção de Data Leakage: Target Encoding (score_diretor, score_estudio, score_ator)
é calculado APENAS no conjunto de treino, após o train_test_split. Os valores são
então aplicados tanto em X_train quanto em X_test (com fallback para a média global).

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
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
# Constantes
# ---------------------------------------------------------------------------
TARGET = 'target_sucesso'

# Features numéricas puras (sem target encoding)
RAW_NUMERIC = [
    'log_budget', 'runtime', 'mes_estreia', 'e_sequencia',
    'e_alta_temporada', 'densidade_investimento',
]

# Colunas categóricas que viram target-encoded scores (apenas no treino)
ENCODING_COLS = ['diretor', 'estudio', 'ator_principal']

# Scores gerados por target encoding (adicionados após o split)
SCORE_COLS = ['score_diretor', 'score_estudio', 'score_ator']

# Feature categórica que vai para OneHotEncoder
CATEGORICAL_FEATURES = ['genero_principal']

# Features numéricas finais para o modelo = brutas + scores
NUMERIC_FEATURES = RAW_NUMERIC + SCORE_COLS

# Colunas lidas do CSV para montar X
FEATURES_CSV = RAW_NUMERIC + ENCODING_COLS + CATEGORICAL_FEATURES

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
assert TARGET in df.columns, f"Coluna '{TARGET}' não encontrada no dataset."

available_csv = [c for c in FEATURES_CSV if c in df.columns]
df_model = df[available_csv + [TARGET]].copy()
df_model.dropna(inplace=True)

X_raw = df_model[available_csv]
y = df_model[TARGET].values

print(f"[2/8] Registros após remoção de NaNs: {len(df_model)}")
print(f"      Features CSV: {available_csv}")
print(f"      Distribuição do target: {dict(zip(*np.unique(y, return_counts=True)))}")

# ---------------------------------------------------------------------------
# 3. Divisão treino/teste
# ---------------------------------------------------------------------------
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[3/8] Divisão 80/20 — Treino: {len(X_train_raw)}, Teste: {len(X_test_raw)}")

# ---------------------------------------------------------------------------
# 3b. Calcular mapas de Target Encoding APENAS no treino (evita Data Leakage)
# ---------------------------------------------------------------------------
y_train_series = pd.Series(y_train, index=X_train_raw.index, name=TARGET)
train_with_target = X_train_raw.join(y_train_series)

score_diretor_map = train_with_target.groupby('diretor')[TARGET].mean().to_dict()
score_estudio_map = train_with_target.groupby('estudio')[TARGET].mean().to_dict()
score_ator_map    = train_with_target.groupby('ator_principal')[TARGET].mean().to_dict()
score_global_media = float(y_train.mean())

print(f"      Score global (média do treino): {score_global_media:.4f}")
print(f"      Diretores mapeados : {len(score_diretor_map)}")
print(f"      Estúdios mapeados  : {len(score_estudio_map)}")
print(f"      Atores mapeados    : {len(score_ator_map)}")

# ---------------------------------------------------------------------------
# 3c. Aplicar scores em treino e teste (unseen → fallback para média global)
# ---------------------------------------------------------------------------
raw_base_cols = [c for c in RAW_NUMERIC + CATEGORICAL_FEATURES if c in X_train_raw.columns]

def apply_scores(X_raw_subset):
    X = X_raw_subset[raw_base_cols].copy()
    X['score_diretor'] = X_raw_subset['diretor'].map(score_diretor_map).fillna(score_global_media)
    X['score_estudio'] = X_raw_subset['estudio'].map(score_estudio_map).fillna(score_global_media)
    X['score_ator']    = X_raw_subset['ator_principal'].map(score_ator_map).fillna(score_global_media)
    return X

X_train = apply_scores(X_train_raw)
X_test  = apply_scores(X_test_raw)

# ---------------------------------------------------------------------------
# 4. Configuração e Treinamento (Pipeline)
# ---------------------------------------------------------------------------
num_features = [f for f in NUMERIC_FEATURES if f in X_train.columns]
cat_features = [f for f in CATEGORICAL_FEATURES if f in X_train.columns]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
])

pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ))
])

print(f"[4/8] Treinando Pipeline (ColumnTransformer -> SMOTE -> Random Forest)...")
pipeline.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 5. Detalhes das transformações
# ---------------------------------------------------------------------------
print(f"[5/8] Detalhes do Processamento:")
print(f"      - ColumnTransformer: StandardScaler ({len(num_features)} features numéricas) + "
      f"OneHotEncoder ({len(cat_features)} features categóricas).")
print(f"      - SMOTE aplicado no treino para balancear classes.")
print(f"      - Random Forest treinado (200 árvores, profundidade ilimitada).")

# ---------------------------------------------------------------------------
# 6. Avaliação
# ---------------------------------------------------------------------------
y_pred = pipeline.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)
precisao = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# CV no conjunto de treino (scores já aplicados — estimativa interna)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1", n_jobs=-1)

print("\n[6/8] Avaliação do modelo (Dados de Teste):")
print(f"      Acurácia  : {acuracia*100:.2f}%")
print(f"      Precisão  : {precisao*100:.2f}%")
print(f"      F1-Score  : {f1*100:.2f}%")
print(f"      Recall    : {recall*100:.2f}%")
print(f"      CV F1 (5-fold, treino): {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
print()
print(classification_report(y_test, y_pred, target_names=["Não Sucesso", "Sucesso"]))

# ---------------------------------------------------------------------------
# 7. Salvar modelo
# ---------------------------------------------------------------------------
os.makedirs(MODELS_DIR, exist_ok=True)

metricas = {
    "acuracia": acuracia,
    "precisao": precisao,
    "f1_score": f1,
    "recall": recall,
    "cv_f1_media": cv_scores.mean(),
    "cv_f1_std": cv_scores.std(),
}

modelo_exportacao = {
    "pipeline": pipeline,
    "features": num_features + cat_features,
    "metricas": metricas,
    "score_diretor_map": score_diretor_map,
    "score_estudio_map": score_estudio_map,
    "score_ator_map": score_ator_map,
    "score_global_media": score_global_media,
}

joblib.dump(modelo_exportacao, MODEL_PATH)
tamanho_kb = os.path.getsize(MODEL_PATH) / 1024
print(f"[7/8] Modelo salvo em: {MODEL_PATH} ({tamanho_kb:.1f} KB)")

# ---------------------------------------------------------------------------
# 8. Gráficos
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Não Sucesso", "Sucesso"]).plot(
    ax=axes[0], cmap="Blues", colorbar=False
)
axes[0].set_title("Matriz de Confusão", fontweight="bold")

# Gráfico 2: Importância de features
rf = pipeline.named_steps['classifier']
ohe_cols = (pipeline.named_steps['preprocessor']
            .named_transformers_['cat']
            .get_feature_names_out(cat_features).tolist())
all_feat_names = num_features + ohe_cols
importances = pd.Series(rf.feature_importances_, index=all_feat_names).sort_values(ascending=False)

top_n = min(15, len(importances))
axes[1].barh(
    importances.index[:top_n][::-1],
    importances.values[:top_n][::-1],
    color='steelblue'
)
axes[1].set_title("Top 15 Features por Importância", fontweight="bold")
axes[1].set_xlabel("Importância (Gini)")

plt.tight_layout()
grafico_path = os.path.join(MODELS_DIR, "avaliacao_modelo.png")
plt.savefig(grafico_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[8/8] Gráfico salvo em: {grafico_path}")

print("\nConcluído com sucesso!")
