"""
Streamlit Dashboard — TMDB 5000 Movies Analysis & Success Prediction

Módulo de Interface Interativa e Visualização.
Consolida os insights da Análise Exploratória (EDA)
e integra a predição gerada pelo pipeline de Machine Learning (Random Forest),
permitindo que o usuário explore métricas financeiras, estatísticas de mercado 
e tendências dinamicamente.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────
# CARREGAMENTO DE DADOS E MODELO EM CACHE
# ─────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    """
    Carrega e pré-processa o dataset TMDB 5000 preparado.
    Garante a integridade dos tipos de dados numéricos e textuais 
    para que a interface e os filtros operem corretamente de forma dinâmica.
    """
    df = pd.read_csv(BASE_DIR / 'data' / 'tmdb_5000_pronto.csv')
    # mes_estreia já é inteiro; garantir tipos numéricos
    df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
    df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')
    df.dropna(subset=['budget', 'runtime'], inplace=True)
    # Preencher nulos nas colunas categóricas para que sorted() funcione
    df['genero_principal'] = df['genero_principal'].fillna('Outros')
    df['estudio']          = df['estudio'].fillna('Independente')
    df['diretor']          = df['diretor'].fillna('Desconhecido')
    # Garantir que e_sequencia seja inteiro (evita conflito no modelo)
    df['e_sequencia'] = df['e_sequencia'].fillna(0).astype(int)
    return df


@st.cache_resource
def load_model():
    """
    Carrega o pipeline estruturado (Transformer, SMOTE e Random Forest)
    treinado e preservado fisicamente na pasta models/.
    """
    return joblib.load(BASE_DIR / 'models' / 'random_forest_model.pkl')


# ─────────────────────────────────────────────────────────────────
# FUNÇÃO DE ANÁLISE COM IA (GROQ)
# ─────────────────────────────────────────────────────────────────

NOMES_MESES = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril",
    5: "Maio", 6: "Junho", 7: "Julho", 8: "Agosto",
    9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
}

@st.cache_data(show_spinner=False)
def gerar_insight_ia(
    genero: str,
    mes: int,
    budget: int,
    sucesso_prob: float,
    acuracia: float,
    f1: float,
) -> str:
    """
    Gera análise textual via Groq LLM com cache por combinação de inputs.

    Parâmetros:
        genero (str): Gênero principal do filme selecionado.
        mes (int): Mês numérico de estreia (1-12).
        budget (int): Orçamento previsto em USD.
        sucesso_prob (float): Probabilidade percentual de sucesso (0-100).
        acuracia (float): Eficiência histórica do modelo (0-100).
        f1 (float): F1-Score histórico do modelo (0-100).

    Retorna:
        str: String contendo a análise textual construída pela IA.
    """
    if not GROQ_API_KEY:
        return "Chave da API Groq não configurada. Verifique o arquivo .env."

    nome_mes = NOMES_MESES.get(mes, str(mes))
    budget_fmt = f"US$ {budget:,.0f}".replace(",", ".")

    prompt = f"""Você é um analista de viabilidade cinematográfica experiente.
Com base nos dados abaixo, forneça uma avaliação objetiva e direta (máximo 4 frases) sobre a viabilidade financeira do filme.

Dados do projeto:
- Gênero: {genero}
- Mês de estreia: {nome_mes}
- Orçamento: {budget_fmt}
- Probabilidade de sucesso prevista pelo modelo: {sucesso_prob:.1f}%
- Modelo utilizado: Random Forest com acurácia de {acuracia:.1f}% e F1-Score de {f1:.1f}%

Estruture sua resposta assim:
1. Avaliação geral do potencial (positiva/negativa/neutra)
2. Contexto do gênero e sazonalidade (mês de estreia)
3. Consideração sobre o orçamento

Seja conciso e use linguagem profissional em português brasileiro."""

    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Erro ao consultar a IA: {str(e)}"


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
st.markdown("Análise de dados de filmes com previsão de sucesso financeiro")

# ─────────────────────────────────────────────────────────────────
# CARREGAR DADOS
# ─────────────────────────────────────────────────────────────────

df = load_data()
model_data = load_model()

score_diretor_map  = model_data.get('score_diretor_map', {})
score_estudio_map  = model_data.get('score_estudio_map', {})
score_ator_map     = model_data.get('score_ator_map', {})
score_global_media = model_data.get('score_global_media', 0.5)

# ─────────────────────────────────────────────────────────────────
# FILTROS DA BARRA LATERAL
# ─────────────────────────────────────────────────────────────────

st.sidebar.header("🔍 Filtros")

# Sequência
only_sequels = st.sidebar.checkbox("Apenas sequências")

# Gêneros
all_genres = sorted(df['genero_principal'].unique())
selected_genres = st.sidebar.multiselect("Gênero", all_genres, default=[])

# Estúdios
top_studios = df['estudio'].value_counts().head(50).index.tolist()
selected_studios = st.sidebar.multiselect("Estúdio", top_studios, default=[])

# Aplicar filtros
filtered_df = df.copy()

if only_sequels:
    filtered_df = filtered_df[filtered_df['e_sequencia'] == 1]

if selected_genres:
    filtered_df = filtered_df[filtered_df['genero_principal'].isin(selected_genres)]

if selected_studios:
    filtered_df = filtered_df[filtered_df['estudio'].isin(selected_studios)]

st.sidebar.markdown(f"**Filmes encontrados:** {len(filtered_df)} / {len(df)}")

# ─────────────────────────────────────────────────────────────────
# ABAS PRINCIPAIS
# ─────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Visão Geral",
    "🎭 Gêneros",
    "🏢 Estúdios",
    "📅 Distribuição por Período",
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
        taxa = filtered_df['target_sucesso'].mean() * 100
        st.metric("✅ Taxa de Sucesso (%)", f"{taxa:.1f}%")
    with col3:
        st.metric("💵 Orçamento Médio", f"${filtered_df['budget'].mean()/1e6:.2f}M")
    with col4:
        st.metric("⏱️ Duração Média", f"{filtered_df['runtime'].mean():.0f} min")

    st.divider()

    # Gráfico de Dispersão: Orçamento × Duração
    st.subheader("Orçamento × Duração por Resultado")
    scatter = px.scatter(
        filtered_df,
        x='budget', y='runtime',
        color=filtered_df['target_sucesso'].map({1: 'Sucesso', 0: 'Não Sucesso'}),
        color_discrete_map={'Sucesso': '#2ecc71', 'Não Sucesso': '#e74c3c'},
        title="Cada ponto: cor = resultado financeiro",
        labels={
            'budget': 'Orçamento (USD)',
            'runtime': 'Duração (min)',
            'color': 'Resultado'
        }
    )
    scatter.update_layout(height=500, template='plotly_white', hovermode='closest')
    st.plotly_chart(scatter, use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# ABA 2: GÊNEROS
# ─────────────────────────────────────────────────────────────────

with tab2:
    st.subheader("Análise por Gênero")

    genre_stats = filtered_df.groupby('genero_principal').agg(
        count=('target_sucesso', 'count'),
        taxa_sucesso=('target_sucesso', 'mean'),
        orcamento_medio=('budget', 'mean'),
        runtime_medio=('runtime', 'mean')
    ).query('count >= 5').sort_values('taxa_sucesso', ascending=True)

    if not genre_stats.empty:
        col1, col2 = st.columns(2)

        # Taxa de Sucesso por gênero
        with col1:
            fig_taxa = px.bar(
                genre_stats,
                x='taxa_sucesso',
                y=genre_stats.index,
                orientation='h',
                color='taxa_sucesso',
                labels={'taxa_sucesso': 'Taxa de Sucesso', 'genero_principal': 'Gênero'},
                title="Taxa de Sucesso por Gênero",
                color_continuous_scale='Turbo'
            )
            fig_taxa.update_layout(height=400, template='plotly_white', showlegend=False)
            fig_taxa.update_xaxes(tickformat='.0%')
            st.plotly_chart(fig_taxa, use_container_width=True)

        # Orçamento Médio por gênero
        with col2:
            genre_orcamento = genre_stats.sort_values('orcamento_medio', ascending=True)
            fig_orc = px.bar(
                genre_orcamento,
                x='orcamento_medio',
                y=genre_orcamento.index,
                orientation='h',
                color='orcamento_medio',
                labels={'orcamento_medio': 'Orçamento Médio (USD)', 'genero_principal': 'Gênero'},
                title="Orçamento Médio por Gênero",
                color_continuous_scale='Viridis'
            )
            fig_orc.update_layout(height=400, template='plotly_white', showlegend=False)
            st.plotly_chart(fig_orc, use_container_width=True)

        # Tabela detalhada
        st.subheader("Estatísticas Detalhadas")
        display_df = genre_stats.copy()
        display_df['taxa_sucesso'] = (display_df['taxa_sucesso'] * 100).round(1)
        display_df['orcamento_medio'] = display_df['orcamento_medio'].round(0)
        display_df['runtime_medio'] = display_df['runtime_medio'].round(1)
        display_df = display_df.sort_values('taxa_sucesso', ascending=False).rename(columns={
            'count': 'Contagem',
            'taxa_sucesso': 'Taxa de Sucesso (%)',
            'orcamento_medio': 'Orçamento Médio (USD)',
            'runtime_medio': 'Duração Média (min)'
        })
        display_df.index.name = 'Gênero'
        st.dataframe(display_df, use_container_width=True)
    else:
        st.warning("Nenhum gênero encontrado com >= 5 filmes nos filtros selecionados.")

# ─────────────────────────────────────────────────────────────────
# ABA 3: ESTÚDIOS
# ─────────────────────────────────────────────────────────────────

with tab3:
    st.subheader("Análise por Estúdio")

    studio_stats = filtered_df.groupby('estudio').agg(
        count=('target_sucesso', 'count'),
        taxa_sucesso=('target_sucesso', 'mean'),
        orcamento_medio=('budget', 'mean')
    ).query('count >= 5').sort_values('taxa_sucesso', ascending=True).tail(10)

    if not studio_stats.empty:
        col1, col2 = st.columns(2)

        # Taxa de Sucesso por estúdio
        with col1:
            fig_studio_taxa = px.bar(
                studio_stats,
                x='taxa_sucesso',
                y=studio_stats.index,
                orientation='h',
                color='taxa_sucesso',
                labels={'taxa_sucesso': 'Taxa de Sucesso', 'estudio': 'Estúdio'},
                title="Top 10 Estúdios — Taxa de Sucesso",
                color_continuous_scale='Reds'
            )
            fig_studio_taxa.update_layout(height=400, template='plotly_white', showlegend=False)
            fig_studio_taxa.update_xaxes(tickformat='.0%')
            st.plotly_chart(fig_studio_taxa, use_container_width=True)

        # Orçamento Médio por estúdio
        with col2:
            studio_orc = studio_stats.sort_values('orcamento_medio', ascending=True)
            fig_studio_orc = px.bar(
                studio_orc,
                x='orcamento_medio',
                y=studio_orc.index,
                orientation='h',
                color='orcamento_medio',
                labels={'orcamento_medio': 'Orçamento Médio (USD)', 'estudio': 'Estúdio'},
                title="Top 10 Estúdios — Orçamento Médio",
                color_continuous_scale='Blues'
            )
            fig_studio_orc.update_layout(height=400, template='plotly_white', showlegend=False)
            st.plotly_chart(fig_studio_orc, use_container_width=True)
    else:
        st.warning("Nenhum estúdio encontrado com >= 5 filmes nos filtros selecionados.")

# ─────────────────────────────────────────────────────────────────
# ABA 4: DISTRIBUIÇÃO POR PERÍODO
# ─────────────────────────────────────────────────────────────────

with tab4:
    st.subheader("Distribuição por Período")

    month_names = {1:'Jan', 2:'Fev', 3:'Mar', 4:'Abr', 5:'Mai', 6:'Jun',
                   7:'Jul', 8:'Ago', 9:'Set', 10:'Out', 11:'Nov', 12:'Dez'}

    col1, col2 = st.columns(2)

    # Taxa de sucesso por mês
    with col1:
        monthly_taxa = (
            filtered_df.groupby('mes_estreia')['target_sucesso']
            .mean()
            .reset_index(name='taxa_sucesso')
        )
        monthly_taxa['mes'] = monthly_taxa['mes_estreia'].map(month_names)

        fig_taxa_mes = px.bar(
            monthly_taxa, x='mes', y='taxa_sucesso',
            title="Taxa de Sucesso por Mês de Estreia",
            labels={'mes': 'Mês', 'taxa_sucesso': 'Taxa de Sucesso'},
            color='taxa_sucesso', color_continuous_scale='Teal'
        )
        fig_taxa_mes.update_layout(height=400, template='plotly_white', showlegend=False)
        fig_taxa_mes.update_yaxes(tickformat='.0%')
        st.plotly_chart(fig_taxa_mes, use_container_width=True)

    # Quantidade de filmes por mês
    with col2:
        monthly_count = (
            filtered_df.groupby('mes_estreia')
            .size()
            .reset_index(name='count')
        )
        monthly_count['mes'] = monthly_count['mes_estreia'].map(month_names)

        fig_count_mes = px.bar(
            monthly_count, x='mes', y='count',
            title="Quantidade de Filmes por Mês de Estreia",
            labels={'mes': 'Mês', 'count': 'Quantidade'},
            color='count', color_continuous_scale='Sunset'
        )
        fig_count_mes.update_layout(height=400, template='plotly_white', showlegend=False)
        st.plotly_chart(fig_count_mes, use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# ABA 5: CORRELAÇÕES
# ─────────────────────────────────────────────────────────────────

with tab5:
    st.subheader("Matriz de Correlação — Pearson")

    numeric_cols = ['budget', 'runtime', 'mes_estreia', 'e_sequencia', 'target_sucesso']
    col_names_pt = {
        'budget': 'Orçamento',
        'runtime': 'Duração',
        'mes_estreia': 'Mês Estreia',
        'e_sequencia': 'É Sequência',
        'target_sucesso': 'Sucesso'
    }

    corr_df = filtered_df[numeric_cols].corr(method='pearson')
    corr_df.rename(columns=col_names_pt, index=col_names_pt, inplace=True)

    fig_corr, ax = plt.subplots(figsize=(8, 6))
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
        annot_kws={'size': 11}
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
    st.subheader("🤖 Previsão de Sucesso Financeiro")

    mets = model_data.get('metricas', {})
    acc  = mets.get('acuracia',  0) * 100
    prec = mets.get('precisao',  0) * 100
    f1   = mets.get('f1_score',  0) * 100
    rec  = mets.get('recall',    0) * 100

    st.info(f"""
    **Como funciona:**
    O modelo classifica se um filme terá **Sucesso Financeiro** —
    definido como receita ≥ 2× o orçamento (receita/orçamento ≥ 2).

    **Performance do modelo treinado:**
    - Acurácia: {acc:.2f}%
    - Precisão: {prec:.2f}%
    - Pontuação F1: {f1:.2f}%
    - Revocação (Recall): {rec:.2f}%
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Entre com os dados do filme")
        budget_input  = st.number_input("Orçamento (USD)", value=50_000_000, step=1_000_000)
        runtime_input = st.number_input("Duração (minutos)", value=120, step=1)
        mes_estreia_input = st.slider("Mês de Estreia", min_value=1, max_value=12, value=6)
        e_sequencia_input = st.checkbox("É uma sequência?")
        genero_input  = st.selectbox("Gênero Principal", sorted(df['genero_principal'].unique()))
        estudio_input = st.selectbox("Estúdio", df['estudio'].value_counts().head(50).index.tolist())
        diretor_input = st.text_input("Diretor", value="Christopher Nolan")
        ator_input    = st.text_input("Ator Principal", value="Robert Downey Jr.")
        threshold = st.slider(
            "Apetite ao Risco (limiar de decisão)",
            min_value=20, max_value=80, value=50, step=5,
            help="Abaixe para prever sucesso com menor confiança necessária."
        ) / 100.0

    with col2:
        st.subheader("🎯 Resultado da Previsão")

        if st.button("🔮 Fazer Previsão", use_container_width=True):
            e_alta_temporada_input = 1 if mes_estreia_input in [6, 7, 12] else 0
            densidade_investimento_input = budget_input / max(runtime_input, 1)
            log_budget_input = np.log1p(budget_input)
            score_dir  = score_diretor_map.get(diretor_input, score_global_media)
            score_est  = score_estudio_map.get(estudio_input, score_global_media)
            score_ator = score_ator_map.get(ator_input, score_global_media)

            X_input = pd.DataFrame([{
                'log_budget':             log_budget_input,
                'runtime':                runtime_input,
                'mes_estreia':            mes_estreia_input,
                'e_sequencia':            int(e_sequencia_input),
                'e_alta_temporada':       e_alta_temporada_input,
                'densidade_investimento': densidade_investimento_input,
                'score_diretor':          score_dir,
                'score_estudio':          score_est,
                'score_ator':             score_ator,
                'genero_principal':       genero_input,
            }])

            ml_pipeline = model_data['pipeline']
            pred_proba   = ml_pipeline.predict_proba(X_input)[0]
            sucesso_prob = pred_proba[1] * 100
            pred_class   = 1 if (sucesso_prob / 100) >= threshold else 0

            # Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=sucesso_prob,
                title={'text': "Probabilidade de Sucesso Financeiro"},
                delta={'reference': threshold * 100},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25],  'color': "#e8f4f8"},
                        {'range': [25, 50], 'color': "#b3d9e6"},
                        {'range': [50, 75], 'color': "#7fb3d5"},
                        {'range': [75, 100],'color': "#2874a6"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold * 100
                    }
                },
                number={'suffix': "%"}
            ))
            fig_gauge.update_layout(height=400)
            st.plotly_chart(fig_gauge, use_container_width=True)

            if pred_class == 1:
                st.success(f"✅ **Sucesso Previsto** ({sucesso_prob:.1f}% de probabilidade)")
            else:
                st.warning(f"⚠️ **Não Sucesso Previsto** ({(100 - sucesso_prob):.1f}% de probabilidade de não sucesso)")

            # ── Insight da IA ──────────────────────────────────────
            st.divider()
            with st.expander("🧠 Insight da IA — Análise do Resultado", expanded=True):
                with st.spinner("Analisando com Inteligência Artificial..."):
                    insight = gerar_insight_ia(
                        genero=genero_input,
                        mes=mes_estreia_input,
                        budget=budget_input,
                        sucesso_prob=sucesso_prob,
                        acuracia=acc,
                        f1=f1,
                    )

                if pred_class == 1:
                    st.success(insight)
                else:
                    st.info(insight)
