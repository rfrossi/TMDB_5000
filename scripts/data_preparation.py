"""
Script de Preparação de Dados para o Dataset TMDB 5000.

Realiza a fusão (merge) entre os diretórios de filmes e seus respectivos elencos.
Trata features em formato JSON (como gêneros principais, produtoras e equipe),
limpa inconsistências (orçamentos ou receitas isolados/zerados) e gera o dataset 
padronizado para a tarefa de Classificação de Sucesso Binário.
O arquivo final de output é `tmdb_5000_pronto.csv`.
"""

import re
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
    """Retorna o primeiro gênero como string simples."""
    genres = parse_json_column(genres_json)
    if isinstance(genres, list) and genres:
        return genres[0].get('name', '')
    return ''

def extract_studio(companies_json):
    """Retorna o primeiro estúdio de produção como string simples."""
    companies = parse_json_column(companies_json)
    if isinstance(companies, list) and companies:
        return companies[0].get('name', '')
    return ''

def extract_director(crew_json):
    """Extrai o nome do Diretor a partir do JSON da equipe (job == 'Director')."""
    crew = parse_json_column(crew_json) if isinstance(crew_json, str) else crew_json
    if isinstance(crew, list):
        for member in crew:
            if member.get('job') == 'Director':
                return member.get('name', '')
    return ''

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

    # 2. Parse das colunas em JSON
    print("  [JSON] Convertendo colunas JSON (genres -> genero_principal)...")
    movies_df['genero_principal'] = movies_df['genres'].apply(extract_genres)

    print("  [JSON] Convertendo colunas JSON (keywords)...")
    movies_df['keywords'] = movies_df['keywords'].apply(parse_json_column)

    print("  [JSON] Convertendo colunas JSON (production_companies -> estudio)...")
    movies_df['estudio'] = movies_df['production_companies'].apply(extract_studio)

    # 3. Substituir zeros por NaN (valores ausentes disfarçados)
    print("  [CLEAN] Substituindo orçamentos zerados por NaN...")
    movies_df['budget'] = movies_df['budget'].replace(0, np.nan)

    print("  [CLEAN] Substituindo receitas zeradas por NaN...")
    movies_df['revenue'] = movies_df['revenue'].replace(0, np.nan)

    print("  [CLEAN] Substituindo runtimes zerados por NaN...")
    movies_df['runtime'] = movies_df['runtime'].replace(0, np.nan)

    # 4. Filtro essencial: descartar linhas sem budget ou revenue (necessários para o target)
    before = len(movies_df)
    movies_df = movies_df.dropna(subset=['budget', 'revenue'])
    after = len(movies_df)
    print(f"  [FILTER] {before - after} linhas descartadas por budget/revenue nulos. Restantes: {after}")

    # 5. Feature Engineering: mês de estreia
    print("  [FEATURE] Extraindo mes_estreia da release_date...")
    movies_df['mes_estreia'] = movies_df['release_date'].dt.month

    # 6. Feature Engineering: identificação de sequência
    print("  [FEATURE] Calculando e_sequencia...")
    def _is_sequel(row):
        # Verifica se 'sequel' está entre as keywords do filme
        keyword_names = [kw.get('name', '').lower() for kw in row['keywords'] if isinstance(kw, dict)]
        if 'sequel' in keyword_names:
            return 1
        # Verifica se o título contém número indicativo de sequência (2, 3, 4...)
        if re.search(r'\b[2-9]\b', str(row.get('title', ''))):
            return 1
        return 0

    movies_df['e_sequencia'] = movies_df.apply(_is_sequel, axis=1)

    # 7. Target: coluna target_sucesso (1 se revenue >= 2 * budget, senão 0)
    print("  [TARGET] Criando coluna target_sucesso (revenue >= 2 * budget)...")
    movies_df['target_sucesso'] = (movies_df['revenue'] >= 2 * movies_df['budget']).astype(int)
    print(f"    [INFO] Distribuição do target: {movies_df['target_sucesso'].value_counts().to_dict()}")

    # 8. Análise informativa de outliers (sem exclusão)
    print("  [INFO] Analisando extremos estatísticos (Blockbusters e Épicos)...")

    runtime_data = movies_df['runtime'].dropna()
    if len(runtime_data) > 0:
        q3_runtime = runtime_data.quantile(0.75)
        upper_bound_rt = q3_runtime + 1.5 * (q3_runtime - runtime_data.quantile(0.25))
        long_movies = (movies_df['runtime'] > upper_bound_rt).sum()
        print(f"    [INFO] {long_movies} filmes com duração > {upper_bound_rt:.0f} min (Mantidos na análise).")

    budget_data = movies_df['budget'].dropna()
    if len(budget_data) > 0:
        q3_budget = budget_data.quantile(0.75)
        upper_bound_bg = q3_budget + 1.5 * (q3_budget - budget_data.quantile(0.25))
        blockbusters = (movies_df['budget'] > upper_bound_bg).sum()
        print(f"    [INFO] {blockbusters} blockbusters com orçamento > ${upper_bound_bg:,.0f} (Mantidos na análise).")

    print(f"[OK] Filmes processados: {len(movies_df)} registros")
    return movies_df

def clean_credits_data(credits_df):
    """Processa e limpa os dados dos créditos de filmes."""
    print("\n[CREDITS] Processando dados de creditos...")

    # Parse de elenco em JSON
    print("  [JSON] Convertendo colunas JSON (cast)...")
    credits_df['cast'] = credits_df['cast'].apply(lambda x: extract_cast(x, limit=10))

    # Ator principal: primeiro nome do elenco
    print("  [JSON] Extraindo ator_principal (primeiro do cast)...")
    credits_df['ator_principal'] = credits_df['cast'].apply(
        lambda c: c[0] if isinstance(c, list) and c else ''
    )

    # Extração do diretor a partir do JSON da equipe
    print("  [JSON] Extraindo diretor da coluna crew...")
    credits_df['diretor'] = credits_df['crew'].apply(extract_director)

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

def add_engineered_features(df):
    """Adiciona features derivadas: sazonalidade, interação e log_budget."""
    print("\n[FEATURE_ENG] Criando features derivadas...")

    # Alta temporada: junho, julho e dezembro
    df['e_alta_temporada'] = df['mes_estreia'].isin([6, 7, 12]).astype(int)
    print("  [OK] e_alta_temporada criada")

    # Densidade de investimento: custo por minuto
    df['densidade_investimento'] = df['budget'] / df['runtime'].replace(0, np.nan)
    print("  [OK] densidade_investimento criada")

    # Log do orçamento: normaliza a escala astronômica de blockbusters
    df['log_budget'] = np.log1p(df['budget'])
    print("  [OK] log_budget criada (np.log1p(budget))")

    # NOTA: Target Encoding (score_diretor, score_estudio, score_ator) é calculado
    # APENAS no conjunto de treino em treinar_modelo.py para evitar Data Leakage.
    # As colunas diretor, estudio e ator_principal são exportadas como texto.

    return df


def validate_data_quality(df):
    """Validação da estrutura e da qualidade dos dados."""
    print("\n[VALIDATE] Validando qualidade dos dados (Pós-Tratamento)...")

    # Colunas finais esperadas no CSV de saída
    final_columns = [
        'budget', 'log_budget', 'runtime', 'genero_principal',
        'estudio', 'diretor', 'ator_principal',
        'mes_estreia', 'e_sequencia',
        'e_alta_temporada', 'densidade_investimento',
        'target_sucesso'
    ]

    for col in final_columns:
        if col in df.columns:
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            print(f"  [NULL] {col}: {missing_pct:.2f}% vazio")
        else:
            print(f"  [WARN] Coluna ausente no dataframe: {col}")

    print("\n  [TYPES] Tipos de dados:")
    for col in ['budget', 'revenue', 'runtime', 'mes_estreia', 'e_sequencia', 'target_sucesso']:
        if col in df.columns:
            print(f"    {col}: {df[col].dtype}")

    return True

def export_prepared_data(df):
    """Exporta apenas as colunas finais para o CSV de classificação binária."""
    print("\n[EXPORT] Exportando dados preparados...")

    final_columns = [
        'budget', 'log_budget', 'runtime', 'genero_principal',
        'estudio', 'diretor', 'ator_principal',
        'mes_estreia', 'e_sequencia',
        'e_alta_temporada', 'densidade_investimento',
        'target_sucesso'
    ]

    # Seleciona apenas as colunas que existem no df (proteção contra ausências)
    cols_to_export = [c for c in final_columns if c in df.columns]
    export_df = df[cols_to_export].copy()

    output_file = DATA_DIR / 'tmdb_5000_pronto.csv'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    export_df.to_csv(output_file, index=False)

    file_size = output_file.stat().st_size / (1024 * 1024)
    print(f"[OK] Arquivo exportado: {output_file} ({file_size:.2f} MB)")
    print(f"     Colunas exportadas: {cols_to_export}")
    print(f"     Total de registros: {len(export_df)}")

    return output_file

def main():
    """Disparo da execução do pipeline estruturado da preparação de base de dados."""
    print("=" * 60)
    print("TASK 2: Preparacao de Dados - TMDB 5000 (Classificacao Binaria)")
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

        # Passo 4b: Adiciona features derivadas (sazonalidade, target encoding, etc.)
        merged_df = add_engineered_features(merged_df)

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