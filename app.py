"""
Streamlit Dashboard — TMDB 5000 Movies Analysis & Popularity Prediction

Módulo de Interface Interativa e Visualização (Card 5).
Consolida os insights da Análise Exploratória (EDA - Card 3)
e integra a predição gerada pelo modelo Random Forest (Card 4),
permitindo que o usuário explore métricas e tendências dinamicamente.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import ast
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────
# CARREGAMENTO DE DADOS E MODELO EM CACHE
# ─────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    """Load and preprocess TMDB 5000 dataset."""
    df = pd.read_csv(BASE_DIR / 'data' / 'tmdb_5000_pronto.csv')

    # Extrair gêneros (string de list -> list)
    df['genres_list'] = df['genres'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    df['main_genre'] = df['genres_list'].apply(lambda lst: lst[0] if lst else 'Unknown')

    # Extrair produtoras cinematográficas
    def extract_company_names(companies_str):
        try:
            if pd.isna(companies_str):
                return []
            comps = ast.literal_eval(companies_str)
            if isinstance(comps, list):
                return [c.get('name', 'Unknown') for c in comps if isinstance(c, dict)]
            return []
        except:
            return []

    df['companies_list'] = df['production_companies'].apply(extract_company_names)
    df['main_company'] = df['companies_list'].apply(lambda lst: lst[0] if lst else 'Unknown')

    # Converter data de lançamento e extrair ano/mês
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year
    df['release_month'] = df['release_date'].dt.month

    # Vamos considerar apenas filmes com orçamento realista (maior que 100 mil dólares)
    df = df[(df['budget'] >= 100000) & (df['revenue'] > 0)].copy()

    # Financial metrics
    df['profit'] = df['revenue'] - df['budget']
    df['roi'] = (df['profit'] / df['budget']) * 100

    return df


@st.cache_resource
def load_model():
    """Load Random Forest ROI pipeline."""
    return joblib.load(BASE_DIR / 'models' / 'random_forest_roi.pkl')


# ─────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TMDB 5000 — Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎬 TMDB 5000 — Dashboard Interativo")
st.markdown("Análise de dados de filmes com previsão de ROI")

# ─────────────────────────────────────────────────────────────────
# CARREGAR DADOS
# ─────────────────────────────────────────────────────────────────

df = load_data()
model_data = load_model()

# ─────────────────────────────────────────────────────────────────
# FILTROS DA BARRA LATERAL
# ─────────────────────────────────────────────────────────────────

st.sidebar.header("🔍 Filtros")

# Intervalo de anos
min_year = int(df['release_year'].min())
max_year = int(df['release_year'].max())
year_range = st.sidebar.slider("Ano de Lançamento", min_year, max_year, (2000, max_year))

# Gêneros
all_genres = sorted(df['main_genre'].unique())
selected_genres = st.sidebar.multiselect("Gênero", all_genres, default=[])

# Estúdios
top_studios = df['main_company'].value_counts().head(50).index.tolist()
selected_studios = st.sidebar.multiselect("Estúdio", top_studios, default=[])

# Aplicar filtros
filtered_df = df[
    (df['release_year'] >= year_range[0]) &
    (df['release_year'] <= year_range[1])
]

if selected_genres:
    filtered_df = filtered_df[filtered_df['main_genre'].isin(selected_genres)]

if selected_studios:
    filtered_df = filtered_df[filtered_df['main_company'].isin(selected_studios)]

st.sidebar.markdown(f"**Filmes encontrados:** {len(filtered_df)} / {len(df)}")

# ─────────────────────────────────────────────────────────────────
# ABAS PRINCIPAIS
# ─────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Visão Geral",
    "🎭 Gêneros",
    "🏢 Estúdios",
    "📈 Tendências",
    "🔗 Correlações",
    "🤖 Previsão ML"
])

# ─────────────────────────────────────────────────────────────────
# ABA 1: VISÃO GERAL
# ─────────────────────────────────────────────────────────────────

with tab1:
    st.subheader("Métricas Principais")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📽️ Total de Filmes", f"{len(filtered_df):,.0f}")
    with col2:
        st.metric("💰 Lucro Médio", f"${filtered_df['profit'].mean()/1e6:.2f}M")
    with col3:
        st.metric("⭐ Nota Média", f"{filtered_df['vote_average'].mean():.2f}/10")
    with col4:
        st.metric("💵 Orçamento Médio", f"${filtered_df['budget'].mean()/1e6:.2f}M")

    st.divider()

    # Gráfico de Dispersão: Orçamento x Receita
    st.subheader("Orçamento × Receita × Popularidade")
    scatter = px.scatter(
        filtered_df,
        x='budget', y='revenue',
        size='popularity', color='main_genre',
        hover_name='title',
        hover_data={'budget': '$,.0f', 'revenue': '$,.0f', 'profit': '$,.0f'},
        title="Cada bolha: tamanho = popularidade, cor = gênero",
        labels={
            'budget': 'Orçamento (USD)', 
            'revenue': 'Receita (USD)',
            'profit': 'Lucro (USD)',
            'popularity': 'Popularidade',
            'main_genre': 'Gênero'
        }
    )
    scatter.update_layout(height=600, template='plotly_white', hovermode='closest')
    st.plotly_chart(scatter, use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# ABA 2: GÊNEROS
# ─────────────────────────────────────────────────────────────────

with tab2:
    st.subheader("Análise por Gênero")

    # Filtrar gêneros com 5 ou mais filmes
    genre_stats = filtered_df.groupby('main_genre').agg({
        'title': 'count',
        'profit': 'mean',
        'roi': 'mean',
        'vote_average': 'mean'
    }).rename(columns={'title': 'count'})

    genre_stats = genre_stats[genre_stats['count'] >= 5].sort_values('profit', ascending=True)

    if not genre_stats.empty:
        col1, col2 = st.columns(2)

        # Lucro médio por gênero
        with col1:
            fig_profit = px.bar(
                genre_stats, x='profit', y=genre_stats.index, orientation='h', color='profit',
                labels={'profit': 'Lucro Médio (USD)', 'main_genre': 'Gênero'},
                title="Lucro Médio por Gênero",
                color_continuous_scale='Turbo'
            )
            fig_profit.update_layout(height=400, template='plotly_white', showlegend=False)
            st.plotly_chart(fig_profit, use_container_width=True)

        # ROI por gênero
        with col2:
            fig_roi = px.bar(
                genre_stats, x='roi', y=genre_stats.index, orientation='h', color='roi',
                labels={'roi': 'Retorno sobre Investimento Médio (%)', 'main_genre': 'Gênero'},
                title="Retorno sobre Investimento Médio por Gênero",
                color_continuous_scale='Viridis'
            )
            fig_roi.update_layout(height=400, template='plotly_white', showlegend=False)
            st.plotly_chart(fig_roi, use_container_width=True)

        # Tabela detalhada
        st.subheader("Estatísticas Detalhadas")
        
        # Display data with translated columns
        display_df = genre_stats.round(2).sort_values('profit', ascending=False).rename(columns={
            'count': 'Contagem',
            'profit': 'Lucro Médio (USD)',
            'roi': 'Retorno sobre Investimento (%)',
            'vote_average': 'Nota Média'
        })
        display_df.index.name = 'Gênero'
        
        st.dataframe(
            display_df,
            use_container_width=True
        )
    else:
        st.warning("Nenhum gênero encontrado com >= 5 filmes nos filtros selecionados.")

# ─────────────────────────────────────────────────────────────────
# ABA 3: ESTÚDIOS
# ─────────────────────────────────────────────────────────────────

with tab3:
    st.subheader("Análise por Estúdio")

    studio_stats = filtered_df.groupby('main_company').agg({
        'title': 'count',
        'profit': ['mean', 'sum'],
        'roi': 'mean'
    }).rename(columns={'title': 'count'})

    studio_stats.columns = ['count', 'profit_mean', 'profit_total', 'roi_mean']
    studio_stats = studio_stats[studio_stats['count'] >= 5].sort_values('profit_total', ascending=True).tail(10)

    if not studio_stats.empty:
        col1, col2 = st.columns(2)

        # Lucro total
        with col1:
            fig_total = px.bar(
                studio_stats, x='profit_total', y=studio_stats.index, orientation='h', color='profit_total',
                labels={'profit_total': 'Lucro Total (USD)', 'main_company': 'Estúdio'},
                title="Top 10 Estúdios — Lucro Total",
                color_continuous_scale='Reds'
            )
            fig_total.update_layout(height=400, template='plotly_white', showlegend=False)
            st.plotly_chart(fig_total, use_container_width=True)

        # ROI médio
        with col2:
            fig_roi_studio = px.bar(
                studio_stats, x='roi_mean', y=studio_stats.index, orientation='h', color='roi_mean',
                labels={'roi_mean': 'Retorno sobre Investimento Médio (%)', 'main_company': 'Estúdio'},
                title="Top 10 Estúdios — Retorno sobre Investimento Médio",
                color_continuous_scale='Blues'
            )
            fig_roi_studio.update_layout(height=400, template='plotly_white', showlegend=False)
            st.plotly_chart(fig_roi_studio, use_container_width=True)
    else:
        st.warning("Nenhum estúdio encontrado com >= 5 filmes nos filtros selecionados.")

# ─────────────────────────────────────────────────────────────────
# ABA 4: TENDÊNCIAS
# ─────────────────────────────────────────────────────────────────

with tab4:
    st.subheader("Tendências Temporais")

    col1, col2 = st.columns(2)

    # Lucro por ano
    with col1:
        yearly = filtered_df.groupby('release_year').agg({
            'profit': 'mean',
            'title': 'count'
        }).rename(columns={'title': 'count', 'profit': 'lucro_medio'})

        fig_year = px.line(
            yearly, x=yearly.index, y='lucro_medio',
            markers=True, title="Lucro Médio por Ano de Lançamento",
            labels={'release_year': 'Ano', 'lucro_medio': 'Lucro Médio (USD)'}
        )
        fig_year.update_layout(height=400, template='plotly_white', hovermode='x unified')
        st.plotly_chart(fig_year, use_container_width=True)

    # Lançamentos por mês
    with col2:
        month_names = {1:'Jan', 2:'Fev', 3:'Mar', 4:'Abr', 5:'Mai', 6:'Jun',
                       7:'Jul', 8:'Ago', 9:'Set', 10:'Out', 11:'Nov', 12:'Dez'}
        monthly = filtered_df.groupby('release_month').size().reset_index(name='count')
        monthly['mes'] = monthly['release_month'].map(month_names)

        fig_month = px.bar(
            monthly, x='mes', y='count',
            title="Quantidade de Filmes por Mês",
            labels={'mes': 'Mês', 'count': 'Quantidade'},
            color='count', color_continuous_scale='Sunset'
        )
        fig_month.update_layout(height=400, template='plotly_white', showlegend=False)
        st.plotly_chart(fig_month, use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# ABA 5: CORRELAÇÕES
# ─────────────────────────────────────────────────────────────────

with tab5:
    st.subheader("Matriz de Correlação — Pearson")

    # Selecionar colunas numéricas
    numeric_cols = ['budget', 'revenue', 'profit', 'roi', 'popularity', 'vote_average', 'vote_count', 'runtime']
    corr_df = filtered_df[numeric_cols].corr(method='pearson')
    
    # Traduzir nomes para o heatmap
    col_names_pt = {
        'budget': 'Orçamento',
        'revenue': 'Receita',
        'profit': 'Lucro',
        'roi': 'Retorno sobre Invest.',
        'popularity': 'Popularidade',
        'vote_average': 'Nota Média',
        'vote_count': 'Qtd. Votos',
        'runtime': 'Duração'
    }
    corr_df.rename(columns=col_names_pt, index=col_names_pt, inplace=True)

    fig_corr, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_df,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.5,
        linecolor='white',
        square=True,
        ax=ax,
        annot_kws={'size': 10}
    )
    ax.set_title("Correlação entre Métricas (Pearson)", fontsize=14, pad=16)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', rotation=0, labelsize=10)
    plt.tight_layout()
    st.pyplot(fig_corr, use_container_width=True)
    plt.close(fig_corr)

    st.info("""
    **Interpretação:**
    - Vermelho intenso: correlação positiva forte (próxima de +1)
    - Azul intenso: correlação negativa forte (próxima de -1)
    - Branco/neutro: sem correlação (próxima de 0)
    """)

# ─────────────────────────────────────────────────────────────────
# ABA 6: PREVISÃO VIA ML
# ─────────────────────────────────────────────────────────────────

with tab6:
    st.subheader("🤖 Previsão de ROI")

    mets = model_data.get('metricas', {})
    r2      = mets.get('r2', 0)
    cv_r2   = mets.get('cv_r2_media', 0)
    cv_std  = mets.get('cv_r2_std', 0)

    st.info(f"""
    **Como funciona:**
    O modelo Random Forest (com PCA) prevê o **ROI esperado** com base exclusivamente em
    variáveis conhecidas **antes** da obra existir — sem receita, nota ou votos (sem data leakage).

    **Features utilizadas:** orçamento (escala log), duração, mês, sazonalidade de férias,
    histórico ROI do diretor, histórico ROI do ator principal, porte do estúdio,
    se é sequência, e {len(model_data.get('top_genres', []))} gêneros em one-hot.

    **Performance — R² (escala log):** `{r2:.4f}` | CV R²: `{cv_r2:.4f} ± {cv_std:.4f}`

    > Quanto mais próximo de 1.0, melhor. O target encoding de diretor/ator é o principal
    > preditor — reflete o histórico de retorno de cada profissional no dataset.
    """)

    # ── Recuperar metadados do modelo ──────────────────────────────
    top_genres      = model_data.get('top_genres', [])
    genre_columns   = model_data.get('genre_columns', [])
    director_enc    = model_data.get('director_encoding', {})
    cast_enc        = model_data.get('cast_encoding', {})
    top_studios     = model_data.get('top_studios', [])
    global_mean_log = model_data.get('global_roi_log_mean', 5.3)
    directors_ui    = ['Novo Diretor (estreante)'] + model_data.get('top_directors_ui', [])
    actors_ui       = ['Sem estrela conhecida'] + model_data.get('top_actors_ui', [])

    month_names = {
        1: 'Janeiro', 2: 'Fevereiro', 3: 'Março', 4: 'Abril',
        5: 'Maio', 6: 'Junho', 7: 'Julho', 8: 'Agosto',
        9: 'Setembro', 10: 'Outubro', 11: 'Novembro', 12: 'Dezembro'
    }

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Planejamento do Filme")

        budget_input  = st.number_input("Orçamento (USD)", value=50_000_000, step=1_000_000, min_value=1)
        runtime_input = st.number_input("Duração (minutos)", value=120, step=1, min_value=1)
        month_input   = st.selectbox(
            "Mês de Estreia",
            options=list(month_names.keys()),
            format_func=lambda m: month_names[m],
            index=5
        )
        genres_input  = st.multiselect(
            "Gênero(s)",
            options=top_genres,
            default=[top_genres[3]] if len(top_genres) > 3 else top_genres[:1]
        )
        director_input = st.selectbox("Diretor", options=directors_ui)
        actor_input    = st.selectbox("Ator Principal", options=actors_ui)
        studio_input   = st.selectbox(
            "Estúdio",
            options=['Estúdio Independente'] + top_studios
        )
        is_sequel_input = st.checkbox("É uma sequência / franquia?")

    with col2:
        st.subheader("🎯 Resultado da Previsão")

        if st.button("🔮 Prever ROI", use_container_width=True):

            # ── Montar vetor de features na ordem exata do treinamento ──
            director_val = director_enc.get(director_input, global_mean_log)
            cast_val     = cast_enc.get(actor_input, global_mean_log)
            top_studio_f = 1 if studio_input in top_studios else 0
            genre_flags  = [1 if g in genres_input else 0 for g in top_genres]

            X_input = np.array([[
                np.log1p(budget_input),         # log_budget
                runtime_input,                   # runtime
                month_input,                     # release_month
                1 if month_input in [6, 7, 12] else 0,  # is_blockbuster_season
                director_val,                    # director_roi_encoded
                cast_val,                        # cast_roi_encoded
                top_studio_f,                    # top_studio
                int(is_sequel_input),            # is_sequel
            ] + genre_flags])

            ml_pipeline  = model_data['pipeline']
            pred_log     = float(ml_pipeline.predict(X_input)[0])
            roi_previsto = float(np.expm1(pred_log) - 100)
            roi_display  = float(np.clip(roi_previsto, -100, 1500))

            bar_color = "#2ecc71" if roi_display >= 0 else "#e74c3c"

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=roi_display,
                title={'text': f"ROI Esperado — {', '.join(genres_input) or '—'} | {month_names[month_input]}"},
                delta={'reference': 0,
                       'increasing': {'color': '#2ecc71'},
                       'decreasing': {'color': '#e74c3c'}},
                gauge={
                    'axis': {'range': [-100, 1500]},
                    'bar': {'color': bar_color},
                    'steps': [
                        {'range': [-100, 0],   'color': "#fadbd8"},
                        {'range': [0, 100],    'color': "#fef9e7"},
                        {'range': [100, 500],  'color': "#d5f5e3"},
                        {'range': [500, 1500], 'color': "#a9dfbf"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 3},
                        'thickness': 0.75,
                        'value': 0
                    }
                },
                number={'suffix': "%", 'valueformat': '.1f'}
            ))
            fig_gauge.update_layout(height=400)
            st.plotly_chart(fig_gauge, use_container_width=True)

            if roi_previsto >= 100:
                st.success(f"✅ **ROI Previsto: {roi_previsto:.1f}%** — Retorno acima do dobro do investimento.")
            elif roi_previsto >= 0:
                st.info(f"ℹ️ **ROI Previsto: {roi_previsto:.1f}%** — Retorno positivo, abaixo de 100%.")
            else:
                st.warning(f"⚠️ **ROI Previsto: {roi_previsto:.1f}%** — Modelo indica risco de prejuízo.")

            lucro_estimado = budget_input * roi_previsto / 100
            st.metric(
                "Lucro Estimado",
                f"${lucro_estimado:,.0f}",
                delta=f"{roi_previsto:.1f}% sobre o orçamento"
            )

            with st.expander("🔍 Detalhes das features utilizadas"):
                st.write({
                    "log_budget":             round(float(np.log1p(budget_input)), 4),
                    "runtime":                runtime_input,
                    "release_month":          month_input,
                    "is_blockbuster_season":  1 if month_input in [6, 7, 12] else 0,
                    "director_roi_encoded":   round(director_val, 4),
                    "cast_roi_encoded":       round(cast_val, 4),
                    "top_studio":             top_studio_f,
                    "is_sequel":              int(is_sequel_input),
                    "genres_selecionados":    genres_input,
                })
