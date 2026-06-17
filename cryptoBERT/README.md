# Análise de Sentimento com CryptoBERT e Web Scrapers

## Visão Geral
Este modulo do projeto tem como objetivo realizar a análise de sentimento do mercado de criptomoedas coletando dados de diversas fontes da internet e processando-os utilizando o **CryptoBERT**, um modelo de linguagem da família BERT pré-treinado e otimizado especificamente para a linguagem, gírias e o jargão do universo cripto.

## Coleta de Dados (Scrapers)
Para alimentar nosso modelo de Machine Learning com as informações mais recentes do mercado, foram desenvolvidos scripts de scraping (raspagem de dados) para extrair textos de relevância para o mercado de criptomoedas de diversas fontes da internet. Dentre as principais estão `Google News`, `Reddit` e `Telegram`, a coleta nessas plataformas é feita de forma diversa (seguindo o fluxo necessário para obter os melhores resultados individuais), mas todas convergem para um resultado comum que serve de input para o nosso modelo de ML.

### Desafios com a Extração de Dados do X (antigo Twitter)
Uma das fontes mais ricas de sentimento imediato e tendências para o mercado de criptomoedas é o X (antigo Twitter). No entanto, enfrentamos desafios significativos na implementação e manutenção da coleta de dados nesta plataforma:
*   **Custo da API Oficial:** A API oficial do X adotou políticas de precificação que se tornaram excessivamente caras para a volumetria de dados contínuos necessária para uma análise abrangente, inviabilizando seu uso no escopo atual da pesquisa.
*   **Limitações de Bibliotecas Alternativas:** Na tentativa de contornar a situação, exploramos o uso de bibliotecas alternativas não-oficiais de scraping, em particular a lib `twikit`. Porém, constatamos que esta biblioteca não estava sendo atualizada com a rapidez e frequência necessárias para acompanhar as constantes e abruptas mudanças no código frontend e nas lógicas de autenticação da plataforma. Isso resultou em quebras frequentes do nosso pipeline de dados, comprometendo a confiabilidade da extração.

Devido a esses impeditivos tecnológicos e financeiros, foi necessário buscar abordagens alternativas para a coleta de dados de redes sociais e diversificar as fontes focando naquelas mais abertas e estáveis (comentadas anteriormente).

## Pipeline de Análise de Sentimento
O núcleo analítico do projeto baseia-se na aplicação do modelo **CryptoBERT** sobre o corpus de texto que construímos.

### Processamento e Análise Geral (JSON)
O processo de consolidação do sentimento do mercado é executado nas seguintes etapas:
1.  **Aglomeração em JSON:** Após a execução bem-sucedida dos scrapers em todas as fontes mapeadas, todos os dados textuais crus coletados são processados, limpos e estruturados em um arquivo `JSON` unificado. Este arquivo atua como a única fonte de verdade de dados para aquele recorte temporal.
2.  **Inferência do CryptoBERT:** Um script itera sobre as entradas deste arquivo JSON, passando cada texto como entrada para o modelo CryptoBERT. O modelo então classifica a polaridade do texto, determinando se o tom da postagem/notícia é *Bullish* (Otimista), *Bearish* (Pessimista) ou *Neutral* (Neutro), junto a isso retorna um score de confiança para essas classes.
3.  **Cálculo do Sentimento Geral:** A análise de sentimento macro do mercado é obtida processando todos os resultados gerados pelo modelo. De maneira fundamental, faz-se o balanço entre os sentimentos extraídos de todo o JSON consolidado para chegar a um veredito sobre o sentimento dominante do mercado no momento.

## Próximos Passos: Refinamento com Pesos e Confiança
Embora o pipeline atual forneça uma boa visão macro, identificamos que tratar todos os dados de texto igualmente pode distorcer a realidade do mercado. O próximo passo fundamental para este projeto é aprofundar-se na metrificação do sentimento introduzindo **sistemas de pesos e fatores de confiança**.

A evolução do cálculo de sentimento geral envolverá:
1.  **Pesos Diferentes para Fontes (Source Weighting):** Nem todas as fontes têm a mesma influência no mercado. Planejamos implementar um sistema onde fontes de alta reputação e impacto no mercado financeiro (ex: sites de notícias especializados e reconhecidos) recebam um multiplicador (peso) maior na equação final de sentimento, enquanto fóruns abertos ou fontes menos verificadas tenham um peso menor.
2.  **Índices de Confiança do Modelo (Confidence Scores):** Além da classificação em si (bullish/bearish), o CryptoBERT fornece a probabilidade daquela predição. O próximo passo envolve utilizar esse valor de confiança para aprimorar o sentimento. Classificações em que o modelo tem alta certeza devem impactar a média muito mais do que classificações limítrofes que beiram a neutralidade.

A integração dessas variáveis permitirá que a ferramenta de análise de sentimento deixe de ser uma média simples e se torne um termômetro muito mais refinado, responsivo e acurado das reais condições do mercado de criptoativos.
