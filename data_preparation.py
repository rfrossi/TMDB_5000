"""
Data Preparation Script for TMDB 5000 Dataset
Task 2: Load, clean, and prepare data for analysis
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path


def load_datasets():
    """Load movies and credits CSV files"""
    print("[LOAD] Carregando datasets...")
    movies_df = pd.read_csv('tmdb_5000_movies.csv')
    credits_df = pd.read_csv('tmdb_5000_credits.csv')
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

    # Parse JSON columns
    print("  [JSON] Convertendo colunas JSON (genres)...")
    movies_df['genres'] = movies_df['genres'].apply(extract_genres)

    print("  [JSON] Convertendo colunas JSON (keywords)...")
    movies_df['keywords'] = movies_df['keywords'].apply(parse_json_column)

    print("  [JSON] Convertendo colunas JSON (production_companies)...")
    movies_df['production_companies'] = movies_df['production_companies'].apply(parse_json_column)

    # Handle budget: convert zeros to NaN
    print("  [CLEAN] Tratando orcamentos zerados...")
    movies_df['budget'] = movies_df['budget'].replace(0, np.nan)

    # Handle revenue: convert zeros to NaN
    print("  [CLEAN] Tratando receitas zeradas...")
    movies_df['revenue'] = movies_df['revenue'].replace(0, np.nan)

    # Handle runtime outliers (remove unrealistic values)
    print("  [CLEAN] Tratando outliers de runtime...")
    q1_runtime = movies_df['runtime'].quantile(0.25)
    q3_runtime = movies_df['runtime'].quantile(0.75)
    iqr_runtime = q3_runtime - q1_runtime
    lower_bound = q1_runtime - 1.5 * iqr_runtime
    upper_bound = q3_runtime + 1.5 * iqr_runtime

    runtime_outliers = movies_df['runtime'].between(lower_bound, upper_bound)
    original_count = len(movies_df)
    movies_df.loc[~runtime_outliers, 'runtime'] = np.nan
    treated_outliers = original_count - runtime_outliers.sum()
    print(f"    [TREATED] {treated_outliers} outliers de runtime")

    # Handle budget outliers
    print("  [CLEAN] Tratando outliers de budget...")
    budget_data = movies_df['budget'].dropna()
    if len(budget_data) > 0:
        q1_budget = budget_data.quantile(0.25)
        q3_budget = budget_data.quantile(0.75)
        iqr_budget = q3_budget - q1_budget
        lower_bound_budget = q1_budget - 1.5 * iqr_budget
        upper_bound_budget = q3_budget + 1.5 * iqr_budget

        budget_outliers = movies_df['budget'].between(lower_bound_budget, upper_bound_budget)
        budget_treated = (~budget_outliers & movies_df['budget'].notna()).sum()
        movies_df.loc[~budget_outliers & movies_df['budget'].notna(), 'budget'] = np.nan
        print(f"    [TREATED] {budget_treated} outliers de budget")

    # Validate and convert data types
    print("  [VALIDATE] Validando tipos de dados...")
    numeric_cols = ['budget', 'revenue', 'runtime', 'popularity', 'vote_average']
    for col in numeric_cols:
        if col in movies_df.columns:
            movies_df[col] = pd.to_numeric(movies_df[col], errors='coerce')

    movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], errors='coerce')

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

    # Merge on movie_id
    merged_df = movies_df.merge(credits_df, on='id', how='left')

    print(f"[OK] Datasets fundidos: {len(merged_df)} registros")
    print(f"    Colunas no dataset: {len(merged_df.columns)}")

    return merged_df


def validate_data_quality(df):
    """Validate data quality"""
    print("\n[VALIDATE] Validando qualidade dos dados...")

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

    output_file = 'tmdb_5000_pronto.csv'
    df.to_csv(output_file, index=False)

    file_size = Path(output_file).stat().st_size / (1024 * 1024)
    print(f"[OK] Arquivo exportado: {output_file} ({file_size:.2f} MB)")

    return output_file


def main():
    """Execute data preparation pipeline"""
    print("=" * 60)
    print("TASK 2: Preparacao de Dados - TMDB 5000")
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
