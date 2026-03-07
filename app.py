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
        st.metric("💵 Budget Médio", f"${filtered_df['budget'].mean()/1e6:.2f}M")

    st.divider()

    # Gráfico de Dispersão: Orçamento x Receita
    st.subheader("Budget × Revenue × Popularidade")
    scatter = px.scatter(
        filtered_df,
        x='budget', y='revenue',
        size='popularity', color='main_genre',
        hover_name='title',
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

    # Recuperando métricas dinâmicas do modelo treinado
    mets = model_data.get('metricas', {})
    mae = mets.get('mae', 0)
    rmse = mets.get('rmse', 0)
    r2 = mets.get('r2', 0)
    cv_r2 = mets.get('cv_r2_media', 0)
    cv_r2_std = mets.get('cv_r2_std', 0)

    st.info(f"""
    **Como funciona:**
    O modelo Random Forest (com PCA) prevê o **ROI esperado** de um filme com base exclusivamente
    em variáveis conhecidas **antes** da obra existir: gênero principal, mês de estreia,
    orçamento e duração.

    Diferente da versão anterior, este modelo **não utiliza receita, nota ou votos** —
    variáveis que só existem após o lançamento — eliminando o vazamento de dados (data leakage).
    O target é transformado em escala logarítmica (log1p) para lidar com a alta variância do ROI.

    **Performance do modelo (escala log, R²):** {r2:.4f}  |  CV R²: {cv_r2:.4f} ± {cv_r2_std:.4f}

    > O R² baixo reflete a alta variabilidade intrínseca do ROI no cinema.
    > A previsão expressa a tendência central histórica para o perfil selecionado.
    """)

    col1, col2 = st.columns(2)

    # Recuperar lista de gêneros do modelo salvo
    genres_list = model_data.get('genres_list', [])
    genre_to_code = model_data.get('genre_to_code', {})

    month_names = {
        1: 'Janeiro', 2: 'Fevereiro', 3: 'Março', 4: 'Abril',
        5: 'Maio', 6: 'Junho', 7: 'Julho', 8: 'Agosto',
        9: 'Setembro', 10: 'Outubro', 11: 'Novembro', 12: 'Dezembro'
    }

    with col1:
        st.subheader("📝 Planejamento do Filme")
        genre_input = st.selectbox(
            "Gênero Principal",
            options=genres_list if genres_list else ['Action', 'Drama', 'Comedy'],
        )
        month_input = st.selectbox(
            "Mês de Estreia",
            options=list(month_names.keys()),
            format_func=lambda m: month_names[m],
            index=5  # Junho como padrão
        )
        budget_input = st.number_input("Orçamento (USD)", value=50_000_000, step=1_000_000, min_value=1)
        runtime_input = st.number_input("Duração (minutos)", value=120, step=1, min_value=1)

    with col2:
        st.subheader("🎯 Resultado da Previsão")

        if st.button("🔮 Prever ROI", use_container_width=True):
            genre_code = genre_to_code.get(genre_input, 0)
            X_input = np.array([[budget_input, runtime_input, month_input, genre_code]])

            ml_pipeline = model_data['pipeline']
            pred_log = float(ml_pipeline.predict(X_input)[0])
            # Inverter transformação log1p(roi+100): expm1(pred) - 100
            roi_previsto = float(np.expm1(pred_log) - 100)
            # Clampar para faixa exibível no gauge (evita valores astronômicos de outliers)
            roi_display = float(np.clip(roi_previsto, -100, 1500))

            # Gauge de ROI — range de -100% a 1500%
            gauge_min, gauge_max = -100, 1500
            bar_color = "#2ecc71" if roi_display >= 0 else "#e74c3c"

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=roi_display,
                title={'text': f"ROI Esperado — {genre_input} | {month_names[month_input]}"},
                delta={'reference': 0, 'increasing': {'color': '#2ecc71'}, 'decreasing': {'color': '#e74c3c'}},
                gauge={
                    'axis': {'range': [gauge_min, gauge_max]},
                    'bar': {'color': bar_color},
                    'steps': [
                        {'range': [-100, 0],    'color': "#fadbd8"},
                        {'range': [0, 100],     'color': "#fef9e7"},
                        {'range': [100, 500],   'color': "#d5f5e3"},
                        {'range': [500, 1500],  'color': "#a9dfbf"}
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

            # Interpretação textual
            if roi_previsto >= 100:
                st.success(f"✅ **ROI Previsto: {roi_previsto:.1f}%** — Retorno acima do dobro do investimento.")
            elif roi_previsto >= 0:
                st.info(f"ℹ️ **ROI Previsto: {roi_previsto:.1f}%** — Retorno positivo, porém abaixo de 100%.")
            else:
                st.warning(f"⚠️ **ROI Previsto: {roi_previsto:.1f}%** — Modelo indica risco de prejuízo.")

            lucro_estimado = budget_input * roi_previsto / 100
            st.metric(
                "Lucro Estimado",
                f"${lucro_estimado:,.0f}",
                delta=f"{roi_previsto:.1f}% sobre o orçamento"
            )
