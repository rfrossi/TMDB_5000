"""
Data Preparation Script for TMDB 5000 Dataset
Task 2: Load, clean, and prepare data for analysis (CORRIGIDO)
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
    """Load movies and credits CSV files"""
    print("[LOAD] Carregando datasets...")
    
    # Lendo os arquivos da pasta 'data'
    movies_df = pd.read_csv(DATA_DIR / 'tmdb_5000_movies.csv')
    credits_df = pd.read_csv(DATA_DIR / 'tmdb_5000_credits.csv')
    
    print(f"[OK] Movies: {len(movies_df)} registros")
    print(f"[OK] Credits: {len(credits_df)} registros")
    return movies_df, credits_df

def parse_json_column(cell):
    """Parse JSON string to list"""
    if pd.isna(cell) or cell == '':
        return []
    try:
        return json.loads(cell)
    except (json.JSONDecodeError, TypeError):
        return []

def extract_genres(genres_json):
    """Extract genre names from JSON"""
    genres = parse_json_column(genres_json)
    return [genre.get('name', '') for genre in genres] if isinstance(genres, list) else []

def extract_cast(cast_json, limit=5):
    """Extract top cast names from JSON"""
    cast = parse_json_column(cast_json)
    if isinstance(cast, list):
        actors = [actor.get('name', '') for actor in cast[:limit]]
        return actors
    return []

def clean_movies_data(movies_df):
    """Clean and process movies data"""
    print("\n[MOVIES] Processando dados dos filmes...")

    # 1. Validar e converter tipos de dados PRIMEIRO (Antes de qualquer cálculo)
    print("  [VALIDATE] Validando e convertendo tipos de dados numéricos e datas...")
    numeric_cols = ['budget', 'revenue', 'runtime', 'popularity', 'vote_average']
    for col in numeric_cols:
        if col in movies_df.columns:
            movies_df[col] = pd.to_numeric(movies_df[col], errors='coerce')
            
    if 'release_date' in movies_df.columns:
        movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], errors='coerce')

    # 2. Parse JSON columns
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

    # 4. Handle Outliers (Apenas identificação estatística, NÃO exclusão)
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

    # Budget info
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
    """Clean and process credits data"""
    print("\n[CREDITS] Processando dados de creditos...")

    # Parse cast JSON
    print("  [JSON] Convertendo colunas JSON (cast)...")
    credits_df['cast'] = credits_df['cast'].apply(lambda x: extract_cast(x, limit=10))

    # Parse crew JSON
    print("  [JSON] Convertendo colunas JSON (crew)...")
    credits_df['crew'] = credits_df['crew'].apply(parse_json_column)

    print(f"[OK] Creditos processados: {len(credits_df)} registros")
    return credits_df

def merge_datasets(movies_df, credits_df):
    """Merge movies and credits by movie_id"""
    print("\n[MERGE] Realizando merge dos datasets...")

    # Rename column for merge
    credits_df = credits_df.rename(columns={'movie_id': 'id'})
    
    # Remove coluna title do credits para não duplicar com o title do movies (title_x, title_y)
    if 'title' in credits_df.columns:
        credits_df = credits_df.drop(columns=['title'])

    # Merge on id
    merged_df = movies_df.merge(credits_df, on='id', how='left')

    print(f"[OK] Datasets fundidos: {len(merged_df)} registros")
    print(f"    Colunas no dataset: {len(merged_df.columns)}")

    return merged_df

def validate_data_quality(df):
    """Validate data quality"""
    print("\n[VALIDATE] Validando qualidade dos dados (Pós-Tratamento)...")

    # Report missing values for key columns
    key_columns = ['id', 'title', 'budget', 'revenue', 'runtime', 'genres', 'cast']
    for col in key_columns:
        if col in df.columns:
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            print(f"  [NULL] {col}: {missing_pct:.2f}% vazio")

    # Report data types
    print("\n  [TYPES] Tipos de dados:")
    for col in ['budget', 'revenue', 'runtime']:
        if col in df.columns:
            print(f"    {col}: {df[col].dtype}")

    return True

def export_prepared_data(df):
    """Export cleaned data to CSV"""
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
    """Execute data preparation pipeline"""
    print("=" * 60)
    print("TASK 2: Preparacao de Dados - TMDB 5000 (Otimizado)")
    print("=" * 60)

    try:
        # Step 1: Load datasets
        movies_df, credits_df = load_datasets()

        # Step 2: Clean movies data
        movies_df = clean_movies_data(movies_df)

        # Step 3: Clean credits data
        credits_df = clean_credits_data(credits_df)

        # Step 4: Merge datasets
        merged_df = merge_datasets(movies_df, credits_df)

        # Step 5: Validate data quality
        validate_data_quality(merged_df)

        # Step 6: Export prepared data
        export_prepared_data(merged_df)

        print("\n" + "=" * 60)
        print("[OK] Pipeline de preparacao concluido com sucesso!")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] Erro durante execucao: {e}")
        raise

if __name__ == "__main__":
    main()