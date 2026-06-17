# Bitscraper: Projeto Final - Tópicos Avançados em IA

## Visão Geral (Macro do Projeto)
Este projeto propõe um **Agente Preditivo Bimodal** para a previsão de preços de criptoativos, combinando séries históricas do mercado financeiro com a análise de sentimento extraída de fontes textuais não estruturadas (notícias e redes sociais da internet). 

O mercado de criptomoedas é altamente volátil e sensível a eventos externos, tornando os modelos determinísticos e de séries temporais tradicionais frequentemente insuficientes. Para superar essa limitação e tratar a incerteza inerente ao sentimento humano de forma matematicamente robusta, nossa arquitetura é dividida em 3 módulos principais integrados:

---

## Módulo 1: Rede Temporal (Bi-LSTM / GRU)
*(A ser desenvolvido)*
- **Objetivo:** Ingerir a série temporal multivariada de dados do mercado (preço de abertura, fechamento, máximas, mínimas, volume e indicadores técnicos).
- **Funcionalidade:** Capturar dependências tanto do passado quanto do contexto recente da série histórica, extraindo padrões não lineares de momentum e volatilidade.

---

## Módulo 2: Processamento de Linguagem Natural e Análise de Sentimento (NLP)

Neste módulo, realizamos a coleta contínua de dados textuais da internet via web scraping (fóruns, notícias e redes sociais) e processamos esses dados para extrair a polaridade de sentimento externo do mercado.

### Por que CryptoBERT ao invés de FinBERT?
A proposta original sugeria o uso do FinBERT. No entanto, embora o FinBERT seja excelente para textos macroeconômicos e do mercado financeiro tradicional, o mercado de criptomoedas possui um ecossistema próprio de linguagem, repleto de jargões, gírias e expressões extremamente específicas (ex: *HODL*, *FUD*, *To the moon*, *Rekt*). 
O **CryptoBERT** foi escolhido como a evolução natural para ser nosso motor de inferência por ser pré-treinado especificamente em textos do universo cripto. Isso garante uma precisão muito superior na interpretação das nuances e assimetrias emocionais desse mercado em particular.

*(Para detalhes aprofundados sobre a arquitetura dos scrapers, dificuldades com a API do X/Twitter e o pipeline de inferência, consulte a documentação dedicada em `cryptoBERT/README.md`).*

### Transformação de Sentimento em Pesos Numéricos para o Motor Fuzzy
Para que o resultado da análise de texto possa alimentar corretamente o motor de decisão do Módulo 3, o sentimento não pode ser entregue de forma puramente categórica ("Bullish", "Bearish", "Neutral"). A lógica neuro-fuzzy exige variáveis de entrada quantificáveis. Portanto, o processo envolve:

1. **Atribuição de Pesos Numéricos:** As classes de sentimento extraídas pelo CryptoBERT são convertidas em valores numéricos (pesos) em uma escala definida (ex: [-1, +1]). 
2. **Índices de Confiança e Fontes:** Incorporamos fatores de ponderação com base na confiança matemática que o modelo teve ao prever aquela classe, bem como a confiabilidade da fonte original da informação.
3. **Consolidação:** A partir de todos os textos de um intervalo de tempo (agregados em JSON), calculamos um escore numérico contínuo final que representa o sentimento geral consolidado daquele momento. Este valor *crisp* (duto/rígido) é a entrada válida essencial para o processo de *fuzzificação* no motor neuro-fuzzy.

---

## Módulo 3: Motor de Decisão Neuro-Fuzzy (ANFIS)
*(A ser desenvolvido)*
- **Objetivo:** Atuar como o núcleo integrador do agente preditivo bimodal.
- **Funcionalidade:** Receber as saídas matemáticas do Módulo 1 (ex: volatilidade, momentum) e do Módulo 2 (escore numérico do sentimento consolidado). Utilizando a inferência fuzzy adaptativa, o sistema mapeará linguisticamente essas incertezas, gerando regras preditivas dinâmicas (no formato "Se-Então") para entregar a previsão final de tendência do criptoativo.