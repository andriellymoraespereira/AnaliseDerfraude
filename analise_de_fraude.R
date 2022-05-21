# Projeto Final 01 - Detecção de Fraudes no Tráfego de Cliques em Propagandas de Aplicações Mobile

# Configurando o diretório de trabalho
setwd("C:/Users/Andrielly/cd/Razure/Projetos-1-2/talkingdata-adtracking-fraud-detection")
getwd()


# Pacotes Necessários para esta análise

install.packages("performanceEstimation")
install.packages("tidyverse")
install.packages("randomForest")
install.packages("caret")
install.packages("lubridade")

# Carrega os pacotes na sessão R
library(tidyverse)
library(randomForest)
library(caret)
library(performanceEstimation)
library(lubridate)

##### Carga dos Dados ##### 

# Carregando os dados
dados <- read.csv("train_sample.csv")

# Dimensões
dim(dados)

# Visualiza os dados
View(dados)

# Variáveis e tipos de dados
str(dados)

# Sumários das variáveis numéricas
summary(dados)


##### Análise Exploratória dos Dados - Limpeza dos Dados ##### 
# Nomes das colunas
colnames(dados)

# Grava os nomes das colunas em um vetor
myColumns <- colnames(dados)
myColumns

# Vamos renomar as colunas para facilitar nosso trabalho mais tarde
myColumns[1] <- "IP_Endereço"
myColumns[2] <- "ID_Aplicativo"
myColumns[3] <- "ID_Dispositivo"
myColumns[4] <- "ID_OS"
myColumns[5] <- "ID_Edições"
myColumns[6] <- "Horário_Fraude"
myColumns[7] <- "Horário_Click_Download_Aplicativo"
myColumns[8] <- "Aplicativo_Baixado"

# Verifica o resultado
myColumns

# Atribui os novos nomes de colunas ao dataframe
colnames(dados) <- myColumns
rm(myColumns)

# Visualiza os dados
View(dados)


# separando as colunas click_time e ttributed_timehora
dados <- dados %>%
  separate(Horário_Fraude , into = c("Ano_Fraude", "Mês_Fraude","Dia_Fraude"), sep = "-")

dados <- dados %>%
  separate(Dia_Fraude , into = c("Dia_Fraude", "Horário"), sep = " ")


dados <- dados %>%
  separate(Horário_Click_Download_Aplicativo , into = c("Ano", "Mês","Dia"), sep = "-")

dados <- dados %>%
  separate(Dia , into = c("Dia", "Tempo"), sep = " ")


# Verificando das colunas com a informação dia do mês.
# Observa-se que existe uma inconsistência nas colunas Dias e Dias Fraudes. Observe que no dia 6 o aplicativo foi baixado 3 vezes, 
# Mas quando analisamos a mesma informação com a coluna Dia_Fraude, o aplicativo foi baixado 5 vezes.

# Cross Tabulation
table(dados$Dia, dados$Aplicativo_Baixado)
table(dados$Dia)
table(dados$Dia_Fraude, dados$Aplicativo_Baixado)
table(dados$Dia_Fraude)


# filter()

dados %>% 
  filter(Aplicativo_Baixado == 1)%>%
  select(Dia_Fraude, Dia,Horário,Tempo, Aplicativo_Baixado) %>%
  arrange(Dia_Fraude) %>% 
  head

# Excluindo as colunas que fazem referencia ao Ano e mês
dados <- select(dados, -Ano_Fraude, -Mês_Fraude, -Dia, -Ano, -Mês,-Tempo,-Dia_Fraude,-Horário)


# Variáveis e tipos de dados
str(dados)

# Convertendo as variaveis em fator (categórica)
dados$IP_Endereço        <- as.numeric(dados$IP_Endereço)
dados$Aplicativo_Baixado <- factor(dados$Aplicativo_Baixado, levels = c(0,1)) 
dados$ID_Aplicativo      <- as.numeric(dados$ID_Aplicativo)
dados$ID_Dispositivo     <- as.numeric(dados$ID_Dispositivo)
dados$ID_OS              <- as.numeric(dados$ID_OS)
dados$ID_Edições         <- as.numeric(dados$ID_Edições)


# Dimensões
dim(dados)

# Sumários das variáveis numéricas
summary(dados)


# Verificando balanceamento do dataset
round(prop.table(table(dados$Aplicativo_Baixado))*100,2)

# Notamos que há um problema nas proporçoes, posteriormente iremos lidar com essa situação.

# Modelo randomForest para criar um plot de importância das variáveis
Modelo <- randomForest(Aplicativo_Baixado ~ . , data = dados, ntree = 100, nodesize = 10, importance = T)
varImpPlot(Modelo)

### Árvores de decisão

#### Dividindo os dados

# Funcao para gerar dados de treino e dados de teste
splitData <- function(dataframe, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  index <- 1:nrow(dataframe)
  trainindex <- sample(index, trunc(length(index)/2))
  trainset <- dataframe[trainindex, ]
  testset <- dataframe[-trainindex, ]
  list(trainset = trainset, testset = testset)
}

# Gerando dados de treino e de teste
splits <- splitData(dados, seed = 808)

# Separando os dados
dados_treino <- splits$trainset
dados_teste <- splits$testset

# Verificando o numero de linhas
nrow(dados_treino)
nrow(dados_teste)

# Construindo o modelo
modelo <- randomForest(Aplicativo_Baixado ~ .
                        -ID_Dispositivo,
                        data = dados_treino, 
                        ntree = 100, 
                        nodesize = 10)

# Imprimondo o resultado
print(modelo)

### Predizendo com o modelo

predict_1 <- predict(modelo,newdata = dados_teste)

# Confusion Matrix
confusionMatrix(predict_1,dados_teste$Aplicativo_Baixado)

# O resultado da acurácia da confusion matrix pode levar a argumentos enganosos. Pois nosso dataset é desbalanceado
# o modelo tende se a viesar para responder quase tudo como a classe de maior proporção, no caso a variavel resposta “1”,
# aplicativo baixados.
# Isso leva a ter um baixo índice de especificidade, na qual é a taxa de acerto do modelo em relação aos verdadeiros negativos.
# No caso em questão, aproximadamente 1,4%. Para contornar esse problema, iremos fazer uma reamostragem do dataset.


## Reamostragem

# Para reamostragem, iremos utilizar o método downsample do pacote caret, na qual irá diminuir o nosso dataset para que as variaveis
# alvo tenham a mesma frequência.
# Concomitantemente iremos usar a técnica smote do pacote performanceEstimation . Nela irá ser criada dados sintéticos para diminuir a diferença de 
# frequência entre os dados da variavel alvo. Essa técninca é baseado no algoritmo KNN, que busca classificar as variaveis com base
# na distancia entre elas.


treino_down <- downSample(dados_treino[,-6],dados_treino[,6])

names(treino_down)[6] <- "Aplicativo_Baixado"

# Smote

treino_smote <- smote(Aplicativo_Baixado ~ .
                      -ID_Dispositivo, data  = dados_treino)  


modelo_down <- train(Aplicativo_Baixado ~ .
                     -ID_Dispositivo,treino_down,method = "rpart",
                     trControl = trainControl(method = "cv"))

modelo_smote <- train(Aplicativo_Baixado ~ .
                      -ID_Dispositivo,treino_smote,method = "rpart",
                      trControl = trainControl(method = "cv"))

# Fazendo Previsões
previ_down <- predict(modelo_down,newdata = dados_teste)
previ_smote <- predict(modelo_smote,newdata = dados_teste)

# Confusion Matrix (Down Sample)
confusionMatrix(previ_down,dados_teste$Aplicativo_Baixado)

# Confusion Matrix (Smote)
confusionMatrix(previ_smote,dados_teste$Aplicativo_Baixado)

### Resultado
# Com o rebalanceamento, a taxa de acurácia cai, porém há um aumento substancial em ambos os casos 
# para a taxa de sensitividade, ou seja, o nosso modelo generaliza melhor para a classe dos verdadeiros negativos.
# Sendo a tecnica smote que apresentou para estes conjunto de dados o melhor resultado para o calculo da Accuracy(0,85) e Specificity (0.90)




# Carregando os dados de teste para fazer predições
teste <- read.csv("test.csv")

# Visualizando os dados
View(teste)


# Nomes das colunas
colnames(teste)

# Grava os nomes das colunas em um vetor
myColumns <- colnames(teste)
myColumns

# Vamos renomar as colunas para facilitar nosso trabalho mais tarde
myColumns[1] <- "ID_previsões"
myColumns[2] <- "IP_Endereço"
myColumns[3] <- "ID_Aplicativo"
myColumns[4] <- "ID_Dispositivo"
myColumns[5] <- "ID_OS"
myColumns[6] <- "ID_Edições"
myColumns[7] <- "Dia_Fraude"

# Atribui os novos nomes de colunas ao dataframe
colnames(teste) <- myColumns
rm(myColumns)


# Verificando se têm dados duplicados coluna IP_Endereço
duplicated(teste$IP_Endereço)

# posição onde está o valor duplicado com "which()"
which(duplicated(teste$IP_Endereço))

# Selecionando valores unicos
teste <- distinct(teste, IP_Endereço, .keep_all = TRUE)


# Convertendo as variaveis em fator (categórica)
teste$IP_Endereço        <- as.numeric(teste$IP_Endereço)
teste$ID_Aplicativo      <- as.numeric(teste$ID_Aplicativo)
teste$ID_Dispositivo     <- as.numeric(teste$ID_Dispositivo)
teste$ID_OS              <- as.numeric(teste$ID_OS)
teste$ID_Edições         <- as.numeric(teste$ID_Edições)

# Excluindo as colunas que fazem referencia ao Ano e mês
teste_2 <- select(teste, -ID_previsões,-Dia_Fraude)


## Prevendo com os dados teste
previ_down_teste <- predict(modelo_down,newdata = teste_2)


# Tabela Final
teste <- select(teste,-Dia_Fraude)
Dados_finais <- cbind(teste,previ_down_teste)
view(Dados_finais)
