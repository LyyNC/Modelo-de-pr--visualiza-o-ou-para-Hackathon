# Modelos-de-previsao-para-Hackathon
Objetivo do Projeto
Este projeto tem como objetivo desenvolver um modelo de previsão de vendas (forecast) para o varejo. A solução prediz a quantidade de vendas semanal por PDV e SKU para as cinco primeiras semanas de janeiro de 2023, utilizando o histórico de vendas de 2022.

💡 Metodologia e Estratégia
Nossa abordagem foi guiada por um pipeline robusto de Ciência de Dados focado em eficiência e escalabilidade, que se mostrou crucial devido ao grande volume de dados, pois, trabalhamos com mais de 200 milhões de linhas.

Amostragem Segura e União dos Dados eram cruciais, para superar os desafios de memória RAM devido ao limite de processamento disponível, a estratégia foi trabalhar todo o arquivo de 6.560.698 linhas da base de transações e uma amostra de mesmo tamanho para os produtos que possui um arquivo maior 195 milhões de linhas, além dos dados de PDV. Os dados de PDV, Transações e de produtos foram filtrados para compor a amostra final, garantindo uma junção eficiente.

Utilizamos ambiente de Linguagem R e Python 3 no google colab, visto que o computador não possuia capacidade, com isso possuía apenas 12GB de RAM para o processamento.
Nos primeiro dias, foi basicamente para encontra um pacote e biblioteca que pudess ler o parquet, como nunca tinhamos utilizado um arquivo tão grande, pocurei alternativas para a obtenção de Dataframe compatível com a memória do ambiente virtual.

Todas quase todas as tentativas foram fracassadas, até que encontrei uma possibilidade utilizando o pacote "duckdb", sendo esse crucial para a manipulação, pois com ele pude utilizar SELECT*FROM, para entrar no ds_produtos e coletar uma amostra de 6.560.698 linhas, e poder seguir com a modelagem.

Após mudança no volume, fizemos novo teste, com excelentes resultados, alé de retirar outlier e padronizar os dados.

Análise Exploratória de Dados (EDA):

Foram criados gráficos para analisar o comportamento dos PDVs, identificando que a maioria das lojas é do tipo Off Premise.
<img width="840" height="840" alt="download" src="https://github.com/user-attachments/assets/f6e66552-3b50-42f6-bdf5-4591b67ce0fa" />

Gráficos de série temporal mostraram uma forte sazonalidade, com quedas de vendas nos finais de semana e feriados, e picos em datas de promoção, além de revelar que as promoções não tiveram um desempenho satisfatório.
<img width="840" height="840" alt="download (14)" src="https://github.com/user-attachments/assets/aaed9d7c-4bec-4a24-86a0-48be996d4ff3" />
<img width="840" height="840" alt="download (12)" src="https://github.com/user-attachments/assets/3457de00-5d9d-4d53-a753-1945cce18ace" />
<img width="840" height="840" alt="download (13)" src="https://github.com/user-attachments/assets/b16cf2b8-33d0-4f7c-aeed-1c2dacff7c6c" />

Durante as visualizações foi verificado que as promoções não tiveram um desempenho esperado então surgiu a hipotese: 
#hipotese de que a queda nas venda em promoção seja pelo fato de que as mesmas
#tenha ocorrido em finais de semanas ou feriados

      promocoes_2022 <- as.Date(c(
        "2022-03-25",
        "2022-07-10",
        "2022-11-25"
      ))
      df_eventos <- vendas_diarias |>
        mutate(is_holiday = ifelse(transaction_date %in% feriados_2022, "Sim", "Não"),
               is_promotion = ifelse(transaction_date %in% promocoes_2022, "Sim", "Não"),
               is_weekend = ifelse(dia_da_semana %in% c(6, 7), "Sim", "Não"))
      analise_eventos <- df_eventos |>
        group_by(is_weekend, is_holiday, is_promotion) |>
        summarise(
          media_vendas = mean(total_quantity, na.rm = TRUE),
          n_dias = n(),
          .groups = "drop"
        )
      
        df_analise_eventos <- analise_eventos |>
        collect()
      
        print(df_analise_eventos, n = Inf)

   A tibble: 6 × 5
  is_weekend is_holiday is_promotion media_vendas n_dias
  <chr>      <chr>      <chr>               <dbl>  <int>
1 Não        Não        Não                90162.    249
2 Sim        Não        Não               295654.    101
3 Não        Sim        Não                99853.      9
4 Sim        Não        Sim                 4391.      1
5 Não        Não        Sim                43426.      2
6 Sim        Sim        Não                 3445.      3

O que foi mostrado que durante a avaliação houve uma queda nas vendas da promoção e uma delas foi em um final de semana.

Essa análise foi a base para a criação de features relevantes.

Engenharia de Variáveis (Feature Engineering):

Os dados foram agregados a um nível de granularidade semanal por PDV e SKU, conforme a necessidade do desafio.
        produtos_unique <- df_produtos_amostra |>
          group_by(produto) |>
          summarise(
            categoria = first(na.omit(categoria)),
            descricao = first(na.omit(descricao)),
            tipos = first(na.omit(tipos)),
            label = first(na.omit(label)),
            subcategoria = first(na.omit(subcategoria)),
            marca = first(na.omit(marca)),
            fabricante = first(na.omit(fabricante)),
            .groups = "drop"
          )

                  df_vendas_enriquecida <- df_vendas_amostra |>
                  left_join(produtos_unique, by = c("internal_product_id" = "produto"))
                        df_vendas_final <- df_vendas_enriquecida |>
                        left_join(df_pdv_amostra, by = c("internal_store_id" = "pdv"))

Foram criadas variáveis de lag (vendas passadas) e media_movel_3 (média móvel de 3 semanas) para capturar a tendência e o comportamento da série temporal.

Foram adicionadas features categóricas (como categoria_pdv e premise) para o modelo.

Utilizamos filtragens para pode encontra valores únicos, e dar mais leveza para a modelagem.

                    pdvs_amostra <- unique(df_vendas_amostra$internal_store_id)
                    ds_pdv <- open_dataset("part-00000-tid-2779033056155408584-f6316110-4c9a-4061-ae48-69b77c7c8c36-4-1-c000.snappy.parquet")
                    df_pdv_amostra <- ds_pdv |>
                      filter(pdv %in% pdvs_amostra) |>
                      collect()|>
                      dplyr::sample_frac()

Modelagem Preditiva:

A escolha do modelo foi o XGBoost, um algoritmo de Machine Learning robusto para dados estruturados, que tem um excelente desempenho em problemas de previsão.

O modelo foi treinado em uma amostra dos dados de 2022, utilizando nrounds=100 e uma divisão de treino e teste para validação.

📊 Análise Visual dos Dados
Aqui estão alguns dos gráficos que criamos para entender melhor os dados:

Distribuição de PDVs por Tipo de Premise

<img width="840" height="840" alt="download (10)" src="https://github.com/user-attachments/assets/0cddbcec-16dc-40f6-a726-a9be98822445" />

Análise: A maioria dos PDVs é do tipo 'Off Premise', indicando um comportamento de vendas dominante nesse segmento.

Quantidade por Categoria PDV (Ordem Decrescente)

<img width="840" height="840" alt="download (9)" src="https://github.com/user-attachments/assets/d65df042-00ee-4162-aff1-a040f5da288c" />

Análise: Este gráfico de barras revela a distribuição da quantidade de PDVs por categoria. Uma análise valiosa para entender as categorias de PDV com maior presença.

Vendas Diárias com Destaque para Finais de Semana

<img width="840" height="840" alt="download (8)" src="https://github.com/user-attachments/assets/02bc2040-5563-4ccf-a51a-cd1e901dd2ab" />

Análise: O gráfico de série temporal mostra que as vendas têm uma forte sazonalidade semanal, com quedas nos finais de semana. Esta é uma feature crucial para o modelo.

Vendas Diárias com Destaque para Feriados

<img width="840" height="840" alt="download (7)" src="https://github.com/user-attachments/assets/9a6b572a-47db-4b58-8bae-4824d3924539" />

Análise: As linhas vermelhas destacam os feriados, mostrando uma queda acentuada nas vendas nesses dias.

Vendas Diárias com Destaque para Promoções

<img width="840" height="840" alt="download (6)" src="https://github.com/user-attachments/assets/2fc06892-1c4f-4fe3-8e51-44a34fb79175" />

Análise: As linhas roxas indicam os dias de promoção, que geralmente levam a picos de vendas.

Vendas por Premise (On/Off)

<img width="840" height="840" alt="download (5)" src="https://github.com/user-attachments/assets/23bc9b05-c6b6-4bd2-b06f-b75db9368090" />

Análise: Este gráfico mostra que o volume de vendas para a premissa 'Off Premise' é significativamente maior do que para 'On Premise'.

Lucro Bruto por Categoria

<img width="840" height="840" alt="download (4)" src="https://github.com/user-attachments/assets/a769c752-324c-45ae-b356-52c3d9f1324d" />

Análise: O lucro bruto por categoria fornece insights sobre a rentabilidade de cada segmento de produto.

Ticket Médio por Premissa

<img width="840" height="840" alt="download (3)" src="https://github.com/user-attachments/assets/d351a4b2-d2a5-49e1-aebf-55f6f17c7d4b" />

Análise: Este gráfico compara o valor médio das transações entre as premissas, mostrando que 'Off Premise' tem um ticket médio mais alto.

Desconto Médio por Categoria de PDV

<img width="840" height="840" alt="download (2)" src="https://github.com/user-attachments/assets/36f2ee76-be6f-487a-9a7b-10eea84c9d02" />

Análise: A visualização do desconto médio por categoria de PDV pode ajudar a identificar padrões de precificação e seu impacto nas vendas.

Evolução Mensal das Vendas por Premissa

<img width="840" height="840" alt="download (1)" src="https://github.com/user-attachments/assets/f45955b1-e7bf-47df-a607-06018261b8e3" />

Análise: Este gráfico de linhas demonstra a tendência das vendas ao longo dos meses para cada tipo de premissa, evidenciando o padrão de sazonalidade anual.

📈 Resultados e Avaliação
A performance do modelo de XGBoost foi avaliada com as seguintes métricas, que indicam um alto poder preditivo na base de validação:

RMSE: 0.48 (Erro médio nas previsões)
 
R²: 0.93 (Poder de explicação do modelo, que é de 93%)

MAE: 0.22  (Erro absoluto médio)


🚀 Como Usar e Pré-requisitos

Pré-requisitos: Instale as bibliotecas R necessárias em seu ambiente (idealmente no Google Colab ou RStudio) usando o seguinte comando:
          install.packages("arrow") #instalando biblioteca de leitura
          install.packages("caret")
          install.packages("xgboost")
          install.packages("tidyverse") #instalando biblioteca de manipulação
          install.packages ("forecast")
          install.packages("duckdb")

Carregando os pacotes para a sessão:
        library(arrow)
        library(dplyr)
        library(lubridate)
        library(duckdb)
        library(zoo)
        library(tidyverse)
        library(ggplot2)
        library(shiny)
        library(forecast)
        library(forcats)
        library(scales)
        library(xgboost)
        library(caret)

Estrutura de Arquivos: Certifique-se de que os arquivos .parquet estejam prontos para utilização.

Execução: O código está contido em um único script, projetado para ser executado em ordem sequencial. 
Ele realizará a amostragem, a união dos dados, a agregação e a modelagem, gerando o arquivo de submissão final.                                                                                                                              

🧑‍💻 Equipe
Izan Cassio Nascimento Pereira

Lincon Souza Pacífico

Pedro Rodrigues Candiani
