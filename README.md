# Previsão de Aluguel de Imóveis no Brasil

**Trabalho prático — Tópicos Avançados em Inteligência Artificial**  
**Professor:** Prof. Dr. Márcio Andrey Teixeira

**Aplicação publicada:** https://previsao-aluguel-ia-zmvuzgk4m3mrjncmcxv6fk.streamlit.app/  
**Código-fonte:** https://github.com/LmZord/previsao-aluguel-ia

## 1. Sobre o projeto

O projeto desenvolve uma aplicação de **aprendizado de máquina supervisionado (regressão)** para estimar o aluguel mensal de imóveis residenciais em cinco cidades brasileiras: **São Paulo, Rio de Janeiro, Belo Horizonte, Porto Alegre e Campinas**.

O usuário informa cidade, área (m²), número de quartos, banheiros e vagas, tipo do imóvel, andar (para apartamentos), valor mensal do condomínio, permissão para animais e presença de mobília. A aplicação retorna o **aluguel mensal estimado em reais (R$)**, o valor somado ao condomínio, o preço estimado por metro quadrado e uma faixa ilustrativa de valores. O resultado pode auxiliar na comparação de anúncios, mas não substitui uma avaliação imobiliária.

## 2. Dataset

Foi utilizado o conjunto de dados **Brazilian Houses to Rent**, disponibilizado no Kaggle, com **10.692 registros de anúncios** de imóveis nas cinco cidades citadas, referentes a **2020**.

- **Fonte:** https://www.kaggle.com/datasets/rubenssjr/brasilian-houses-to-rent
- **Arquivo utilizado:** `dataset/houses_to_rent_v2.csv`
- **Variável-alvo:** valor mensal do aluguel (`rent amount (R$)`, renomeado para `aluguel`).
- **Principais atributos de entrada:** cidade, área, quartos, banheiros, vagas, andar, condomínio, aceitação de animais e mobília.

Durante a preparação, foram removidas duplicatas e registros com valores extremos considerados inconsistentes, o andar de casas/térreos foi codificado como zero e foram criados atributos derivados. As colunas de seguro-incêndio e valor total foram descartadas para evitar **vazamento da resposta**, pois dependem do aluguel; o IPTU também foi excluído da modelagem.

## 3. Treinamento e escolha do modelo

O treinamento foi realizado no **Google Colab**, utilizando Python e Scikit-learn. Os dados foram separados em **80% para treinamento e 20% para teste** (semente aleatória 42). A seleção de configurações utilizou validação cruzada com **5 divisões no conjunto de treinamento**, mantendo o teste separado da escolha do modelo.

Foram comparadas três alternativas:

1. **Regressão Linear**, com transformação logarítmica do aluguel;
2. **Árvore de Decisão**, com ajuste de hiperparâmetros;
3. **Random Forest**, também com ajuste de hiperparâmetros.

As variáveis categóricas foram codificadas, e o pré-processamento foi integrado ao pipeline para aplicar as transformações de forma compatível no treinamento e na inferência. O critério principal de seleção foi o **menor erro absoluto médio (MAE) na validação cruzada**.

O modelo selecionado e publicado foi o **Random Forest**. As métricas exibidas na aplicação para o conjunto de teste são:

| Métrica | Resultado |
|---|---:|
| Erro absoluto médio (MAE) | Aproximadamente **R$ 1.372** |
| Erro percentual mediano | Aproximadamente **27%** |
| Coeficiente de determinação (R²) | **0,617** |

A comparação completa, incluindo RMSE, gráficos e análise dos erros, encontra-se no notebook de treinamento. As métricas indicam que o modelo oferece estimativas aproximadas, não preços garantidos.

## 4. Aplicação e funcionamento

A interface foi desenvolvida em **Streamlit** e publicada no **Streamlit Community Cloud**. A aplicação carrega o modelo previamente salvo em `modelo/modelo_aluguel.pkl` e executa uma nova previsão para cada consulta, **sem realizar novo treinamento**.

O formulário valida informações obrigatórias e entradas inválidas, como áreas inferiores a 10 m². No caso de **casa/térreo**, a aplicação usa internamente `andar = 0`; para apartamentos, o andar deve ser informado.

A tela de resultados apresenta o aluguel estimado, o valor somado ao condomínio, o valor por metro quadrado e uma faixa **ilustrativa** calculada a partir do erro percentual mediano. Essa faixa não é um intervalo estatístico de confiança nem uma garantia de preço.

**Acessar a aplicação:** https://previsao-aluguel-ia-zmvuzgk4m3mrjncmcxv6fk.streamlit.app/

## 5. Estrutura do repositório

```text
previsao-aluguel-ia/
├── app.py                           # Interface e inferência no Streamlit
├── requirements.txt                 # Dependências da aplicação e versões
├── README.md                        # Documentação do projeto
├── dataset/
│   └── houses_to_rent_v2.csv        # Dataset utilizado
├── modelo/
│   └── modelo_aluguel.pkl           # Modelo treinado e pré-processamento
└── notebook/
    └── Treinamento_Aluguel.ipynb    # Preparação, treinamento e avaliação
```

## 6. Como executar

### Utilizar a aplicação publicada

Abra o [site no Streamlit](https://previsao-aluguel-ia-zmvuzgk4m3mrjncmcxv6fk.streamlit.app/), preencha os campos e clique em **Estimar aluguel**. Não é necessário instalar programas ou criar uma conta.

### Executar localmente

Recomenda-se utilizar **Python 3.12**, compatível com as versões registradas no projeto.

```bash
git clone https://github.com/LmZord/previsao-aluguel-ia.git
cd previsao-aluguel-ia
python -m venv .venv
```

Ative o ambiente virtual:

- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **Linux/macOS:** `source .venv/bin/activate`

Instale as dependências e inicie a aplicação:

```bash
pip install -r requirements.txt
streamlit run app.py
```

O arquivo `modelo/modelo_aluguel.pkl` já está incluído no repositório; **não é necessário treinar o modelo novamente** para executar a aplicação.

### Reproduzir o treinamento no Google Colab

1. Abra `notebook/Treinamento_Aluguel.ipynb` no Google Colab.
2. Execute as células em sequência, autorizando a montagem do Google Drive quando solicitado.
3. O notebook obtém ou reutiliza o dataset, prepara os dados, compara os algoritmos e apresenta as métricas e os gráficos de avaliação.
4. Os arquivos gerados são salvos, conforme configurado no notebook, em `Meu Drive/Lab_IA/Projeto_Aluguel/`, incluindo `modelo/modelo_aluguel.pkl` e `requirements.txt`.
5. Caso faça novo treinamento e deseje publicar a nova versão, atualize esses arquivos no repositório, preservando seus caminhos e verificando a compatibilidade das dependências.

O notebook utiliza também bibliotecas de visualização, como Matplotlib e Seaborn, presentes normalmente no ambiente Colab.

## 7. Testes realizados

Foram executados **cinco casos de teste** na aplicação publicada: três entradas válidas, uma inválida e uma situação limítrofe. A documentação da entrega contém capturas de tela dos dados preenchidos e dos resultados.

| Teste | Cenário | Resultado observado |
|---|---|---|
| 1 | Apartamento de 35 m² em São Paulo | Aluguel estimado: **R$ 2.329** |
| 2 | Apartamento de 90 m² em Belo Horizonte | Aluguel estimado: **R$ 2.049** |
| 3 | Casa/térreo de 200 m² em Porto Alegre | Aluguel estimado: **R$ 6.766** |
| 4 | Imóvel com área igual a 0 m² | Entrada recusada: **“A área precisa ter pelo menos 10 m².”** |
| 5 | Apartamento de 900 m² no Rio de Janeiro | Aluguel estimado: **R$ 9.623** |

O caso 5 demonstra a necessidade de cautela com imóveis de características atípicas. Os resultados são **previsões**, e não valores reais comprovados de mercado. Para a verificação de reprodutibilidade, uma mesma entrada deve gerar resultados compatíveis no código de inferência do notebook e na aplicação publicada.

## 8. Limitações

- Os anúncios são de **2020**, sem atualização monetária automática ou correção pela inflação.
- O modelo contempla somente as **cinco cidades** presentes no dataset.
- Não são considerados bairro, estado de conservação, vista, localização exata nem infraestrutura de lazer do edifício.
- Imóveis de alto padrão ou muito diferentes dos exemplos usados no treinamento podem apresentar erros maiores.
- A faixa apresentada na interface é **ilustrativa**, não constituindo garantia de preço ou intervalo de confiança calibrado.

## 9. Integrantes e contribuições

- **Lucas Marcelino:** Responsável pelo desenvolvimento e implementação técnica do projeto, incluindo a organização e adaptação dos códigos de treinamento e da aplicação, integração do modelo de Machine Learning à interface Streamlit, configuração do repositório no GitHub, gerenciamento das dependências e publicação da aplicação no Streamlit Community Cloud. Também responsável pela realização dos testes funcionais, identificação e correção de erros e verificação do funcionamento da solução.
- **Ruan Vinicius Almagro:** Responsável pela pesquisa e seleção do dataset utilizado, considerando sua adequação ao problema de previsão de aluguéis. Participou da definição do problema e dos objetivos do projeto, do levantamento das informações sobre os dados e da elaboração da documentação acadêmica, incluindo a descrição da metodologia, interpretação dos resultados, análise das limitações e organização das referências bibliográficas.

## 10. Uso de ferramentas de IA

O **ChatGPT (OpenAI)** foi utilizado como ferramenta de apoio à revisão da documentação, orientação sobre integração e publicação no Streamlit, identificação e correção de problemas na interface e organização dos casos de teste. O treinamento e a avaliação do modelo foram executados no Google Colab; a aplicação e seus resultados foram submetidos a testes manuais.

## 11. Referências

- **Kaggle — Brazilian Houses to Rent:** https://www.kaggle.com/datasets/rubenssjr/brasilian-houses-to-rent
- **Scikit-learn:** https://scikit-learn.org/stable/
- **Streamlit:** https://docs.streamlit.io/
- **Pandas:** https://pandas.pydata.org/docs/
- **NumPy:** https://numpy.org/doc/
- **Google Colab:** https://colab.research.google.com/
