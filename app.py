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
        vazios = [nome for nome, valor in obrigatorios.items() if valor is None]
    if vazios:
        erros.append("Preencha os campos: " + ", ".join(vazios) + ".")
    if area is not None and area < 10:
        erros.append("A área precisa ter pelo menos 10 m².")
    if banheiros == 0:
        erros.append("Informe pelo menos 1 banheiro.")
    if area and quartos and area / max(quartos, 1) < 6:
        erros.append("Área muito pequena para esse número de quartos — confira os valores.")

    # Fora da faixa dos dados de treino: permite, mas avisa
    if area is not None and area > faixas["area"][1]:
        avisos.append(f"Área acima de {faixas['area'][1]:.0f} m²: o modelo viu poucos imóveis assim.")
    if condominio is not None and condominio > faixas["condominio"][1]:
        avisos.append("Condomínio acima do maior valor visto no treino: estimativa menos confiável.")
    for campo, valor in [("quartos", quartos), ("banheiros", banheiros), ("vagas", vagas)]:
        if valor is not None and valor > faixas[campo][1]:
            avisos.append(f"Nº de {campo} acima do máximo visto no treino ({faixas[campo][1]:.0f}).")

    if erros:
        for e in erros:
            st.error(e)
        st.stop()

    # -----------------------------------------------------------------------
    # 7. Inferência (mesmo processamento do treino)
    # -----------------------------------------------------------------------
    entrada = pd.DataFrame([{
        "cidade": cidade, "area": int(area), "quartos": int(quartos), "banheiros": int(banheiros),
        "vagas": int(vagas), "andar": 0 if tipo == "Casa / térreo" else int(andar),
        "animal": "acept" if animal == "Sim" else "not acept",
        "mobiliado": "furnished" if mobiliado == "Sim" else "not furnished",
        "condominio": float(condominio),
    }])
    previsao = float(modelo.predict(criar_atributos(entrada))[0])

    # -----------------------------------------------------------------------
    # 8. Resultado em linguagem simples
    # -----------------------------------------------------------------------
    for a in avisos:
        st.warning(a)
    erro_pct = metricas["Erro % mediano"] / 100
    st.subheader("Resultado")
    c1, c2 = st.columns(2)
    c1.metric("Aluguel estimado", f"R$ {previsao:,.0f}".replace(",", "."))
    c2.metric("Aluguel + condomínio", f"R$ {previsao + condominio:,.0f}".replace(",", "."))
    faixa_min, faixa_max = previsao * (1 - erro_pct), previsao * (1 + erro_pct)
    # Exibição monetária em métricas evita que o Streamlit interprete "$" como LaTeX.
    st.write("Faixa estimada de preços para um imóvel com essas características:")
    col_min, col_max = st.columns(2)
    col_min.metric("Limite inferior", f"R$ {faixa_min:,.0f}".replace(",", "."))
    col_max.metric("Limite superior", f"R$ {faixa_max:,.0f}".replace(",", "."))
    st.caption(
        f"A faixa apresentada é uma estimativa ilustrativa baseada no desempenho do modelo e não representa uma garantia do valor real de mercado."
    )
    st.write(f"Valor por m²: **R$ {previsao / area:,.2f}**".replace(",", "X").replace(".", ",").replace("X", "."))
    st.info("Se o anúncio que você está avaliando está **bem acima da faixa**, ele pode estar caro — ou ter "
            "algo que o modelo não conhece (bairro nobre, reforma, lazer completo).")
    with st.expander("Ver dados enviados ao modelo"):
        st.dataframe(entrada, hide_index=True)

# ---------------------------------------------------------------------------
# 9. Limitações
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "**Limitações:** baseado em anúncios de 2020 (sem correção pela inflação), apenas 5 cidades e sem "
    "informação de bairro, conservação ou lazer do prédio — fatores que mudam muito o preço. O valor é uma "
    "estimativa estatística, não uma avaliação imobiliária."
)
