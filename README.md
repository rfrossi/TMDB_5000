# TMDB 5000 - Movie Data Analysis

Projeto de análise de dados do dataset TMDB 5000 utilizando Python, com foco em preparação e exploração de dados para modelagem de inteligência artificial.

## Dataset

O projeto utiliza o dataset [TMDB 5000 Movies](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) obtido do Kaggle.

### Instruções para Download do Dataset
1. Acesse o [link oficial no Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).
2. Faça o download do arquivo compactado.
3. Extraia os arquivos `tmdb_5000_movies.csv` e `tmdb_5000_credits.csv`.
4. Salve ambos os arquivos dentro da pasta `data/` na raiz deste projeto.

## Requisitos

- Python 3.13+
- Poetry 2.0+ (para gerenciamento de dependências)

## Instalação

### 1. Clonar o repositório

```bash
git clone https://github.com/rfrossi/TMDB_5000.git
cd TMDB_5000
```

### 2. Instalar dependências com Poetry

```bash
poetry install
```

Isso criará automaticamente um ambiente virtual em `.venv/` com todas as dependências especificadas no `pyproject.toml`.

### 3. Ativar o ambiente virtual

**Opção 1: Usar Poetry**
```bash
poetry shell
```

**Opção 2: Usar o interpretador do VS Code**
- Abra o projeto em VS Code
- Pressione `Ctrl+Shift+P` (ou `Cmd+Shift+P` no Mac)
- Digite "Python: Select Interpreter"
- Escolha o caminho `./.venv/bin/python`

## Dependências Principais

- **pandas** (2.3.3) - Manipulação e análise de dados
- **numpy** (2.4.2) - Computação numérica
- **scikit-learn** (1.8.0) - Machine Learning
- **matplotlib** (3.10.8) - Visualização de dados
- **seaborn** (0.13.2) - Visualização estatística
- **streamlit** (1.55.0) - Criação de aplicações web
- **jupyterlab** (4.5.5) - Notebooks interativos
- **ipykernel** (7.2.0) - Kernel para Jupyter

## Estrutura do Projeto

```
TMDB_5000/
├── README.md              # Este arquivo
├── pyproject.toml         # Configuração do Poetry e dependências
├── poetry.lock            # Lock das versões exatas das dependências
├── .gitignore             # Arquivos ignorados pelo Git
├── .venv/                 # Ambiente virtual (ignorado pelo Git)
├── referencias/           # Arquivos de referência e modelos
└── notebooks/             # Notebooks Jupyter (em desenvolvimento)
```

## Uso

### Jupyter Lab

Para iniciar um Jupyter Lab:

```bash
poetry run jupyter lab
```

### Preparar os dados e treinar o modelo

Antes de executar o painel, é necessário gerar os artefatos (`data/tmdb_5000_pronto.csv` e `models/random_forest_model.pkl`), que não estão no repositório:

```bash
# 1. Preparar os dados brutos (gera data/tmdb_5000_pronto.csv)
poetry run python scripts/data_preparation.py

# 2. Treinar o modelo (gera models/random_forest_model.pkl)
poetry run python scripts/treinar_modelo.py
```

### Streamlit

Com os artefatos gerados, execute o painel interativo:

```bash
poetry run streamlit run app.py
```

## Desenvolvimento

### Adicionar novas dependências

```bash
poetry add <package-name>
```

### Atualizar dependências

```bash
poetry update
```

### Remover dependências

```bash
poetry remove <package-name>
```

## Principais Insights ("A Fórmula do Sucesso")

Nosso relatório analítico completo pode ser lido em [INSIGHTS.md](./INSIGHTS.md). De maneira resumida, identificamos que a "Fórmula do Sucesso" hegemônica prioriza a mitigação de risco:
- **Alto Orçamento vs Risco Estético:** Investimentos gigantescos estão associados a gêneros familiares e franquias, visando lucros globais e licenciamentos, em vez de excelência artística/narrativa.
- **Terror como Refúgio:** Filmes de Horror apresentam ROI gigantesco, sendo um formato financeiramente sustentável e psicologicamente instigante sem precisar de estrelas milionárias.
- **Engajamento Supera Qualidade:** O modelo preditivo provou que a quantidade de avaliações (*vote_count*) é um preditor muito mais forte que a nota em si (*vote_average*). Ou seja, filmes polarizadores que geram debates massivos nas redes são mais rentáveis que obras elogiadas, mas silenciosas.
Para saber mais e entender todas as limitações analíticas do dataset, confira o [INSIGHTS.md](./INSIGHTS.md).

## Referências

Consulte o script `scripts/data_preparation.py` para detalhes sobre a abordagem de preparação de dados.

## Licença

MIT
