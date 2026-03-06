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

    # Métricas financeiras
    df['profit'] = df['revenue'] - df['budget']
    df['roi'] = np.where(df['budget'] > 0, (df['profit'] / df['budget']) * 100, 0)

    # Filtrar para dados financeiros válidos
    df = df[(df['budget'] > 0) & (df['revenue'] > 0)].copy()

    return df


@st.cache_resource
def load_model():
    """Load Random Forest pipeline."""
    return joblib.load(BASE_DIR / 'models' / 'random_forest_model.pkl')


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
st.markdown("Análise de dados de filmes com previsão de popularidade")

# ─────────────────────────────────────────────────────────────────
# CARREGAR DADOS
# ─────────────────────────────────────────────────────────────────

df = load_data()
pipeline = load_model()

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
        st.metric("💵 Budget Médio", f"${filtered_df['budget'].mean()/1e6:.2f}M")

    st.divider()

    # Gráfico de Dispersão: Orçamento x Receita
    st.subheader("Budget × Revenue × Popularidade")
    scatter = px.scatter(
        filtered_df,
        x='budget', y='revenue',
        size='popularity', color='main_genre',
        hover_name='title_x',
        hover_data={'budget': '$,.0f', 'revenue': '$,.0f', 'profit': '$,.0f'},
        title="Cada bolha: tamanho = popularidade, cor = gênero",
        labels={'budget': 'Budget (USD)', 'revenue': 'Receita (USD)'}
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
        'title_x': 'count',
        'profit': 'mean',
        'roi': 'mean',
        'vote_average': 'mean'
    }).rename(columns={'title_x': 'count'})
    genre_stats = genre_stats[genre_stats['count'] >= 5].sort_values('profit', ascending=True)

    if not genre_stats.empty:
        col1, col2 = st.columns(2)

        # Lucro médio por gênero
        with col1:
            fig_profit = px.bar(
                genre_stats, x='profit', y=genre_stats.index, orientation='h', color='profit',
                labels={'profit': 'Lucro Médio (USD)'},
                title="Lucro Médio por Gênero",
                color_continuous_scale='Turbo'
            )
            fig_profit.update_layout(height=400, template='plotly_white', showlegend=False)
            st.plotly_chart(fig_profit, use_container_width=True)

        # ROI por gênero
        with col2:
            fig_roi = px.bar(
                genre_stats, x='roi', y=genre_stats.index, orientation='h', color='roi',
                labels={'roi': 'ROI Médio (%)'},
                title="ROI Médio por Gênero",
                color_continuous_scale='Viridis'
            )
            fig_roi.update_layout(height=400, template='plotly_white', showlegend=False)
            st.plotly_chart(fig_roi, use_container_width=True)

        # Tabela detalhada
        st.subheader("Estatísticas Detalhadas")
        st.dataframe(
            genre_stats.round(2).sort_values('profit', ascending=False),
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
        'title_x': 'count',
        'profit': ['mean', 'sum'],
        'roi': 'mean'
    }).rename(columns={'title_x': 'count'})
    studio_stats.columns = ['count', 'profit_mean', 'profit_total', 'roi_mean']
    studio_stats = studio_stats[studio_stats['count'] >= 5].sort_values('profit_total', ascending=True).tail(10)

    if not studio_stats.empty:
        col1, col2 = st.columns(2)

        # Lucro total
        with col1:
            fig_total = px.bar(
                studio_stats, x='profit_total', y=studio_stats.index, orientation='h', color='profit_total',
                labels={'profit_total': 'Lucro Total (USD)'},
                title="Top 10 Estúdios — Lucro Total",
                color_continuous_scale='Reds'
            )
            fig_total.update_layout(height=400, template='plotly_white', showlegend=False)
            st.plotly_chart(fig_total, use_container_width=True)

        # ROI médio
        with col2:
            fig_roi_studio = px.bar(
                studio_stats, x='roi_mean', y=studio_stats.index, orientation='h', color='roi_mean',
                labels={'roi_mean': 'ROI Médio (%)'},
                title="Top 10 Estúdios — ROI Médio",
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
            'title_x': 'count'
        }).rename(columns={'title_x': 'count', 'profit': 'lucro_medio'})

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

    fig_corr = px.imshow(
        corr_df,
        color_continuous_scale='RdBu_r', color_continuous_midpoint=0,
        zmin=-1, zmax=1,
        title="Correlação entre Métricas (Pearson)",
        labels={'color': 'Correlação'},
        aspect='auto'
    )
    fig_corr.update_layout(height=600, width=800)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.info("""
    **Interpretação:**
    - Vermelho: correlação positiva (aumentam juntas)
    - Azul: correlação negativa (variam inversamente)
    - Branco: sem correlação
    """)

# ─────────────────────────────────────────────────────────────────
# ABA 6: PREVISÃO VIA ML
# ─────────────────────────────────────────────────────────────────

with tab6:
    st.subheader("🤖 Previsão de Popularidade")

    st.info("""
    **Como funciona:**
    O modelo Random Forest foi treinado com 3,229 filmes para prever se um filme
    terá **alta popularidade** (acima da mediana) ou **baixa popularidade**.

    **Performance do modelo:**
    - Acurácia: 86.53%
    - Precisão: 86.20%
    - F1-Score: 86.59%
    - Recall: 87.00%
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Entre com os dados do filme")
        budget_input = st.number_input("Budget (USD)", value=50e6, step=1e6)
        revenue_input = st.number_input("Receita (USD)", value=150e6, step=1e6)
        runtime_input = st.number_input("Duração (minutos)", value=120, step=1)
        vote_avg_input = st.number_input("Nota Média (0-10)", value=7.0, step=0.1, min_value=0.0, max_value=10.0)
        vote_count_input = st.number_input("Quantidade de Votos", value=1000, step=100)

    with col2:
        st.subheader("🎯 Resultado da Previsão")

        if st.button("🔮 Fazer Previsão", use_container_width=True):
            # Preparar entrada
            X_input = np.array([[budget_input, revenue_input, runtime_input, vote_avg_input, vote_count_input]])

            # Obter componentes do modelo
            scaler = pipeline['scaler']
            pca = pipeline['pca']
            model = pipeline['model']

            # Escalonamento de variáveis
            X_scaled = scaler.transform(X_input)

            # PCA - Redução de Componentes
            X_pca = pca.transform(X_scaled)

            # Realizar a predição
            pred_proba = model.predict_proba(X_pca)[0]
            pred_class = model.predict(X_pca)[0]

            # Obter informações de predição
            mediana_pop = pipeline.get('mediana_popularidade', 11.16)
            alta_prob = pred_proba[1] * 100  # Probability of high popularity (class 1)

            # Exibição gráfica gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=alta_prob,
                title={'text': "Probabilidade de Alta Popularidade"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "#e8f4f8"},
                        {'range': [25, 50], 'color': "#b3d9e6"},
                        {'range': [50, 75], 'color': "#7fb3d5"},
                        {'range': [75, 100], 'color': "#2874a6"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                },
                number={'suffix': "%"}
            ))
            fig_gauge.update_layout(height=400)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Interpretação de resultado
            if pred_class == 1:
                st.success(f"✅ **Alta Popularidade Prevista** ({alta_prob:.1f}% de confiança)")
            else:
                st.warning(f"⚠️ **Baixa Popularidade Prevista** ({(100-alta_prob):.1f}% de confiança)")

            st.metric("Mediana de Popularidade (Treino)", f"{mediana_pop:.2f}")
