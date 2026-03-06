# TMDB 5000 - Movie Data Analysis

Projeto de análise de dados do dataset TMDB 5000 utilizando Python, com foco em preparação e exploração de dados para modelagem de inteligência artificial.

## Dataset

O projeto utiliza o dataset [TMDB 5000 Movies](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) obtido do Kaggle.

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

### Streamlit

Para executar uma aplicação Streamlit:

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

## Referências

Consulte o notebook de referência em `referencias/DataPrep_luiscarlos_VERSAO_FINAL.ipynb` para mais detalhes sobre a abordagem de preparação de dados.

## Licença

MIT
