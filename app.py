"""
Aplicativo Streamlit — Estimativa do valor do aluguel no Brasil
Trabalho prático 1 — Tópicos Avançados em Inteligência Artificial

Executar localmente:  streamlit run app.py
Arquivos necessários:  app.py, requirements.txt, modelo/modelo_aluguel.pkl (gerado pelo notebook)
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# 1. Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Estimativa de Aluguel", page_icon="🏠", layout="centered")
CAMINHO_MODELO = "modelo/modelo_aluguel.pkl"


# ---------------------------------------------------------------------------
# 2. Carregamento do modelo (uma vez; cada consulta só faz a inferência)
# ---------------------------------------------------------------------------
@st.cache_resource
def carregar_modelo():
    return joblib.load(CAMINHO_MODELO)


try:
    pacote = carregar_modelo()
except FileNotFoundError:
    st.error(f"Arquivo do modelo não encontrado em '{CAMINHO_MODELO}'. Execute o notebook e envie o .pkl.")
    st.stop()

modelo, faixas, metricas = pacote["modelo"], pacote["faixas"], pacote["metricas_teste"]


# ---------------------------------------------------------------------------
# 3. Mesma função de atributos derivados do notebook (precisa ser IDÊNTICA)
# ---------------------------------------------------------------------------
def criar_atributos(df):
    df = df.copy()
    df["casa_terreo"] = (df["andar"] == 0).astype(int)
    df["log_area"] = np.log(df["area"])
    return df


# ---------------------------------------------------------------------------
# 4. Cabeçalho
# ---------------------------------------------------------------------------
st.title("🏠 Quanto deveria custar este aluguel?")
st.write(
    "Informe as características do imóvel e o modelo estima o **valor mensal do aluguel** com base em "
    "mais de 10 mil anúncios de São Paulo, Rio de Janeiro, Belo Horizonte, Porto Alegre e Campinas. "
    "Use para avaliar se um anúncio está caro ou barato, ou para definir o preço do seu imóvel."
)

with st.sidebar:
    st.header("Sobre o modelo")
    st.write(f"**Algoritmo:** {pacote['nome_modelo']}")
    st.write("**Desempenho no conjunto de teste:**")
    st.write(f"- Erro absoluto médio (MAE): R$ {metricas['MAE']:,.0f}")
    st.write(f"- Erro percentual mediano: {metricas['Erro % mediano']:.0f}%")
    st.write(f"- R²: {metricas['R²']:.3f}")
    st.caption("Dataset: Brazilian Houses to Rent (Kaggle) — anúncios de 2020.")

# ---------------------------------------------------------------------------
# 5. Formulário (campos começam vazios para permitir a validação)
# ---------------------------------------------------------------------------
with st.form("formulario"):
    cidade = st.selectbox("Cidade", pacote["cidades"], index=None, placeholder="Selecione a cidade")
    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input("Área (m²)", min_value=0, max_value=5000, value=None, step=5, placeholder="ex.: 70")
        quartos = st.number_input("Quartos", min_value=0, max_value=20, value=None, step=1, placeholder="ex.: 2")
        banheiros = st.number_input("Banheiros", min_value=0, max_value=20, value=None, step=1, placeholder="ex.: 1")
        vagas = st.number_input("Vagas de garagem", min_value=0, max_value=20, value=None, step=1, placeholder="ex.: 1")
    with col2:
        tipo = st.radio("Tipo", ["Apartamento", "Casa / térreo"], horizontal=True)
        andar = st.number_input("Andar", min_value=1, max_value=60, value=None, step=1, placeholder="ex.: 5",
                                disabled=False, help="Ignorado para casa/térreo.")
        condominio = st.number_input("Condomínio (R$/mês)", min_value=0, max_value=50000, value=None, step=50,
                                     placeholder="0 se não houver")
        animal = st.radio("Aceita animais?", ["Sim", "Não"], horizontal=True)
        mobiliado = st.radio("Mobiliado?", ["Não", "Sim"], horizontal=True)
    enviar = st.form_submit_button("Estimar aluguel", type="primary")

# ---------------------------------------------------------------------------
# 6. Validação das entradas
# ---------------------------------------------------------------------------
if enviar:
    erros, avisos = [], []
    obrigatorios = {"Cidade": cidade, "Área": area, "Quartos": quartos, "Banheiros": banheiros,
                    "Vagas": vagas, "Condomínio": condominio}
    if tipo == "Apartamento":
        obrigatorios["Andar"] = andar
