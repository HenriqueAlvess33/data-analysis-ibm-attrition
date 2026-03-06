# 📊 Análise e Previsão de Turnover – IBM HR Attrition

Este repositório faz parte de um projeto em duas partes dedicado à análise de rotatividade de funcionários (*turnover*) com base no clássico dataset **IBM HR Employee Attrition**.  

Os dois repositórios compartilham o mesmo objetivo – entender e prevenir a saída de talentos – mas abordam o problema de formas complementares:

| Repositório | Foco | Tecnologias |
|-------------|------|-------------|
| **1. EDA Interativa** | Análise exploratória visual e interativa dos dados | Streamlit, Seaborn, Matplotlib, Pandas |
| **2. Modelo Preditivo** | Aplicação de Machine Learning para classificação de risco individual | Streamlit, Scikit‑learn / PyCaret, Joblib |

> 💡 **Links:**  
> - [Repositório de EDA](https://github.com/HenriqueAlvess33/data-analysis-ibm-attrition/tree/main)  
> - [Repositório do Modelo Preditivo](https://github.com/HenriqueAlvess33/ibm_attrition_model)  

---

## 📁 Dataset

O dataset utilizado é o famoso **IBM HR Analytics Employee Attrition & Performance**, disponível publicamente no [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset). Ele contém dados fictícios de 1470 funcionários e 35 variáveis, incluindo informações demográficas, satisfação no trabalho, métricas de desempenho e, claro, a coluna alvo **`Attrition`** (se o funcionário saiu ou não da empresa).

Em ambos os repositórios, o arquivo `HR-Employee-Attrition.csv` está incluso para facilitar a execução imediata (no repositório de EDA) ou o treinamento do modelo (no repositório preditivo).

---

## 🚀 Funcionalidades dos Aplicativos

### 1. Aplicativo de Análise Exploratória (EDA)

Este app foi construído com **Streamlit** e oferece recursos para explorar os dados de forma dinâmica:

- **Carregamento e configuração inicial** – upload de CSV, seleção da coluna alvo e classificação inteligente das variáveis (categóricas vs. numéricas) com limiar de cardinalidade ajustável.
- **Seleção flexível de variáveis** – escolha quais colunas analisar.
- **Visualizações para variáveis categóricas** – gráficos de barras com proporção de turnover por categoria.
- **Visualizações para variáveis numéricas** – dois modos: gráficos absolutos (boxplots/violinplots) e gráficos normalizados (KDE + histograma de proporções), além de análise de risco por quartil.
- **Insights pré‑calculados** – painel lateral com observações resumidas sobre grupos de risco.

### 2. Aplicativo de Previsão (Modelo Preditivo)

Este app também é desenvolvido em **Streamlit** e permite prever a probabilidade de um funcionário deixar a empresa com base em um modelo treinado.

- **Formulário completo** – preencha dados pessoais, profissionais, de satisfação e remuneração.
- **Avatares de exemplo** – carregue rapidamente perfis pré‑definidos (Claudio, Henrique, Zélia, etc.) para testar o modelo.
- **Cálculo automático de `MonthlyRate`** – como esta variável não é preenchida pelo usuário, o app a estima usando a média por cargo (carregada de um arquivo JSON).
- **Predição em tempo real** – ao clicar em "Confirmar e gerar previsão", o modelo carregado (Naive Bayes treinado com PyCaret) retorna a classe prevista (`Yes`/`No`) e a probabilidade associada.
- **Transparência dos dados** – um expansor mostra exatamente quais valores foram usados na predição.

---

## 🛠️ Como executar localmente (qualquer um dos repositórios)

1. **Clone o repositório desejado**:
   ```bash
   git clone https://github.com/seu-usuario/analise-turnover-eda.git   # ou o repositório de ML
   cd nome-do-repositorio
   ```
2. **Instale as dependências** (comum aos dois apps):
   ```bash
   pip install streamlit pandas numpy matplotlib seaborn pillow scikit-learn joblib
   ```
   *Se houver um arquivo `requirements.txt`, utilize‑o.*

3. **Execute o aplicativo**:
   - Para o app de EDA:
     ```bash
     streamlit run 1_eda_app.py
     ```
   - Para o app de previsão:
     ```bash
     streamlit run 2_predicao_app.py
     ```
   *(Ajuste os nomes dos scripts conforme necessário.)*

4. **Acesse** no navegador: `http://localhost:8501`

---

## Acessando página via navegador

**Render:** Host online com planos gratuitos, utilizado neste projeto para que o usuário não necessite executar o script localmente

**Link para EDA Interativo:** https://data-analysis-ibm-attrition.onrender.com/

**Link para Modelo Predítivo:** https://ibm-attrition-model.onrender.com

---

## 📦 Estrutura dos repositórios

### Repositório de EDA
```
├── data_analysis.py            # Script principal da análise exploratória
├── HR-Employee-Attrition.csv   # Dataset incluso
├── images/                     # Imagens utilizadas na interface (ex: office.jpg, ibm_logo.png)
├── requirements.txt            # (opcional) Lista de dependências
└── README.md                   # Este arquivo
```

### Repositório do Modelo Preditivo
```
├── model_applying.py                       # Script principal da previsão
├── modelo_naive_bayes_02_02_2026.pkl       # Modelo treinado 
├── media_monthly_rate_per_job_role.json    # Média de MonthlyRate por cargo (fallback)
├── creating_model.ipynb                    # Notebook para criação do modelo
├── images/                                 # Imagens utilizadas (IBM_image.jpg, snapchat-circle.png)
├── HR-Employee-Attrition.csv               # Dataset (opcional, para referência)
├── requirements.txt
└── README.md
```

---

## 🤖 Sobre o modelo preditivo

O modelo utilizado no segundo app foi treinado pelo notebook creating_modelo.ipynb, que se valeu da lógiva empreada nas bibliotecas **PyCaret** usando o mesmo dataset IBM. Após comparação de diversos algoritmos, o **Naive Bayes** apresentou o melhor equilíbrio entre desempenho e simplicidade. O pipeline completo (incluindo pré‑processamento) foi salvo com `joblib`.

A variável `MonthlyRate` não é solicitada no formulário porque seu valor é inferido a partir da média do cargo – um tratamento feito durante o treinamento para evitar vazamento de dados (*data leakage*).

---

## ✍️ Autor

Desenvolvido por [Henrique Alves](https://github.com/HenriqueAlvess33) como parte de um estudo aprofundado sobre rotatividade de funcionários.
