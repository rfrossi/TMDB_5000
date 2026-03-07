"""
Script de Preparação de Dados para o Dataset TMDB 5000.
Responsável pelo Card 2: 
Realiza a fusão (merge) entre os filmes e seus respectivos elencos, 
trata features JSON (como gêneros, companhias e cast),
limpa inconsistências (orçamento/receita zerados) e gera os dados padronizados 
para EDA e machine learning (tmdb_5000_pronto.csv). (CORRIGIDO PARA IGNORAR OUTLIERS)
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path

# Configuração de Diretórios usando pathlib
# .parent.parent assume que o script está em uma subpasta (ex: /src/scripts/script.py)
# Se o script estiver na raiz do projeto (mesmo nível da pasta data), mude para apenas .parent
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'

def load_datasets():
    """Carrega origens CSV de filmes e créditos."""
    print("[LOAD] Carregando datasets...")
    
    # Lendo os arquivos da pasta 'data'
    movies_df = pd.read_csv(DATA_DIR / 'tmdb_5000_movies.csv')
    credits_df = pd.read_csv(DATA_DIR / 'tmdb_5000_credits.csv')
    
    print(f"[OK] Movies: {len(movies_df)} registros")
    print(f"[OK] Credits: {len(credits_df)} registros")
    return movies_df, credits_df

def parse_json_column(cell):
    """Converte strings baseadas em JSON em listas."""
    if pd.isna(cell) or cell == '':
        return []
    try:
        return json.loads(cell)
    except (json.JSONDecodeError, TypeError):
        return []

def extract_genres(genres_json):
    """Extração de nomes de gêneros a partir do JSON."""
    genres = parse_json_column(genres_json)
    return [genre.get('name', '') for genre in genres] if isinstance(genres, list) else []

def extract_cast(cast_json, limit=5):
    """Extração de atores principais do elenco via JSON."""
    cast = parse_json_column(cast_json)
    if isinstance(cast, list):
        actors = [actor.get('name', '') for actor in cast[:limit]]
        return actors
    return []

def clean_movies_data(movies_df):
    """Processamento e limpeza da estrutura dos filmes."""
    print("\n[MOVIES] Processando dados dos filmes...")

    # 1. Validar e converter tipos de dados PRIMEIRO (Antes de qualquer cálculo)
    print("  [VALIDATE] Validando e convertendo tipos de dados numéricos e datas...")
    numeric_cols = ['budget', 'revenue', 'runtime', 'popularity', 'vote_average']
    for col in numeric_cols:
        if col in movies_df.columns:
            movies_df[col] = pd.to_numeric(movies_df[col], errors='coerce')
            
    if 'release_date' in movies_df.columns:
        movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], errors='coerce')

    # Parse das colunas em JSON
    print("  [JSON] Convertendo colunas JSON (genres)...")
    movies_df['genres'] = movies_df['genres'].apply(extract_genres)

    print("  [JSON] Convertendo colunas JSON (keywords)...")
    movies_df['keywords'] = movies_df['keywords'].apply(parse_json_column)

    print("  [JSON] Convertendo colunas JSON (production_companies)...")
    movies_df['production_companies'] = movies_df['production_companies'].apply(parse_json_column)

    # 3. Handle missing data (zeros that should be NaN)
    print("  [CLEAN] Substituindo orçamentos zerados por NaN...")
    movies_df['budget'] = movies_df['budget'].replace(0, np.nan)

    print("  [CLEAN] Substituindo receitas zeradas por NaN...")
    movies_df['revenue'] = movies_df['revenue'].replace(0, np.nan)
    
    print("  [CLEAN] Substituindo runtimes zerados por NaN...")
    movies_df['runtime'] = movies_df['runtime'].replace(0, np.nan)

    # Tratar outliers de tempo de duração do filme (Apenas identificação estatística, NÃO exclusão)
    print("  [INFO] Analisando extremos estatísticos (Blockbusters e Épicos)...")
    
    # Runtime info
    runtime_data = movies_df['runtime'].dropna()
    if len(runtime_data) > 0:
        q1_runtime = runtime_data.quantile(0.25)
        q3_runtime = runtime_data.quantile(0.75)
        iqr_runtime = q3_runtime - q1_runtime
        upper_bound_rt = q3_runtime + 1.5 * iqr_runtime
        long_movies = (movies_df['runtime'] > upper_bound_rt).sum()
        print(f"    [INFO] {long_movies} filmes possuem duração superior a {upper_bound_rt:.0f} min (Mantidos na análise).")

    # Avaliar e tratar outliers da métrica orçamentária
    budget_data = movies_df['budget'].dropna()
    if len(budget_data) > 0:
        q1_budget = budget_data.quantile(0.25)
        q3_budget = budget_data.quantile(0.75)
        iqr_budget = q3_budget - q1_budget
        upper_bound_bg = q3_budget + 1.5 * iqr_budget
        blockbusters = (movies_df['budget'] > upper_bound_bg).sum()
        print(f"    [INFO] {blockbusters} filmes são blockbusters com orçamento > ${upper_bound_bg:,.0f} (Mantidos na análise).")

    print(f"[OK] Filmes processados: {len(movies_df)} registros")
    return movies_df

def clean_credits_data(credits_df):
    """Processa e limpa os dados dos créditos de filmes."""
    print("\n[CREDITS] Processando dados de creditos...")

    # Parse de elenco em JSON
    print("  [JSON] Convertendo colunas JSON (cast)...")
    credits_df['cast'] = credits_df['cast'].apply(lambda x: extract_cast(x, limit=10))

    # Parse da equipe no formato JSON
    print("  [JSON] Convertendo colunas JSON (crew)...")
    credits_df['crew'] = credits_df['crew'].apply(parse_json_column)

    print(f"[OK] Creditos processados: {len(credits_df)} registros")
    return credits_df

def merge_datasets(movies_df, credits_df):
    """Cria junção das bases de filmes e creditos pela ID."""
    print("\n[MERGE] Realizando merge dos datasets...")

    # Renomear referência id para fundir em seguida
    credits_df = credits_df.rename(columns={'movie_id': 'id'})
    
    # Remove coluna title do credits para não duplicar com o title do movies (title_x, title_y)
    if 'title' in credits_df.columns:
        credits_df = credits_df.drop(columns=['title'])

    # Fusão referencial pela coluna movie_id
    merged_df = movies_df.merge(credits_df, on='id', how='left')

    print(f"[OK] Datasets fundidos: {len(merged_df)} registros")
    print(f"    Colunas no dataset: {len(merged_df.columns)}")

    return merged_df

def validate_data_quality(df):
    """Validação da estrutura e da qualidade dos dados."""
    print("\n[VALIDATE] Validando qualidade dos dados (Pós-Tratamento)...")

    # Retorna contagem de métricas vitais ausentes
    key_columns = ['id', 'title', 'budget', 'revenue', 'runtime', 'genres', 'cast']
    for col in key_columns:
        if col in df.columns:
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            print(f"  [NULL] {col}: {missing_pct:.2f}% vazio")

    # Relatório de checagem dos tipos de colunas do Df
    print("\n  [TYPES] Tipos de dados:")
    for col in ['budget', 'revenue', 'runtime']:
        if col in df.columns:
            print(f"    {col}: {df[col].dtype}")

    return True

def export_prepared_data(df):
    """Exporta os dataframes higienizados para o arquivo de destino em csv."""
    print("\n[EXPORT] Exportando dados preparados...")

    # Salvando o arquivo de volta na pasta 'data'
    output_file = DATA_DIR / 'tmdb_5000_pronto.csv'
    
    # Cria a pasta data caso ela não exista ainda
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_file, index=False)
    
    # Usando stat() do pathlib para pegar o tamanho do arquivo
    file_size = output_file.stat().st_size / (1024 * 1024)
    print(f"[OK] Arquivo exportado: {output_file} ({file_size:.2f} MB)")

    return output_file

def main():
    """Disparo da execução do pipeline estruturado da preparação de base de dados."""
    print("=" * 60)
    print("TASK 2: Preparacao de Dados - TMDB 5000 (Otimizado)")
    print("=" * 60)

    try:
        # Passo 1: Carrega datasets iniciais
        movies_df, credits_df = load_datasets()

        # Passo 2: Limpa os dados em movies
        movies_df = clean_movies_data(movies_df)

        # Passo 3: Prepara as colunas em credits
        credits_df = clean_credits_data(credits_df)

        # Passo 4: Agrega todos datasets correlacionados
        merged_df = merge_datasets(movies_df, credits_df)

        # Passo 5: Avaliar validação e logar os valores e qualidades dos dados nulos
        validate_data_quality(merged_df)

        # Passo 6: Cria exportação da rotina preparada para csv final
        export_prepared_data(merged_df)

        print("\n" + "=" * 60)
        print("[OK] Pipeline de preparacao concluido com sucesso!")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] Erro durante execucao: {e}")
        raise

if __name__ == "__main__":
    main()