
#Manipulação de dados
import pandas as pd
import numpy as np

#Manipulação datas
from datetime import datetime
from dateutil import parser
from datetime import datetime

### Traduzindo noticias
from googletrans import Translator

#### Features Roberta
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

### Features FinBERT
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

import itertools
from google.oauth2 import service_account
from google.cloud import bigquery
import pandas_gbq
import json

service_account_info = json.load(open('stack-minio-1c0130346112.json'))
credentials = service_account.Credentials.from_service_account_info(service_account_info)
client = bigquery.Client(credentials=credentials)

#petro
query_job = client.query("SELECT * FROM `stack-minio.dts_stack_minio.raw_yfinance`")
df_petro = query_job.result().to_dataframe()
df_petro.Date = pd.to_datetime(df_petro.Date).dt.date
df_petro.drop(['Open', 'High','Low','Close'], axis=1, inplace = True)
df_petro['Var%'] = df_petro['Adj_Close'].pct_change()
df_petro = df_petro[df_petro['Var%'].notna()]
df_petro.drop_duplicates(inplace=True)
df_petro.to_csv('df_petro.csv', index = False, sep='|')

# gnews
query_job = client.query("SELECT * FROM `stack-minio.dts_stack_minio.raw_googlenews`")
df = query_job.result().to_dataframe()
df.date = pd.to_datetime(df.date, format="%d/%m/%Y").dt.date
df.drop_duplicates(inplace=True)
df.to_csv('df_gnews.csv', index = False, sep='|')

## Gerando uma lista com todos os dias:
start_date = '01/01/2020'
end_date = '31/12/2022'

#Transformando para o padrão inglês
start_date = datetime.strptime(start_date, '%d/%m/%Y').strftime('%m-%d-%Y')
end_date = datetime.strptime(end_date, '%d/%m/%Y').strftime('%m-%d-%Y')

#Gerando a lista com todas as datas
todas_datas = pd.date_range(start=start_date, end=end_date, freq = '1D')
todas_datas = [i.strftime("%d/%m/%Y") for i in todas_datas ]

##Datas
datas = df.date.value_counts()  
data_df = datas.reset_index()
data_df
data_df['index'] = pd.to_datetime(data_df['index'])
data_df.columns = ['Datas', 'Num_Noticias']
data_df['Mes'] = data_df['Datas'].dt.month

#Gerando lista com todas as datas com noticias
datas_com_noticias = [i.strftime("%d/%m/%Y") for i in data_df['Datas'] ]

#Gerando lista com todas as datas sem noticias em 2021
datas_sem_noticias = [i for i in todas_datas if i not in datas_com_noticias]

datas_com_pregao = [i.strftime("%d/%m/%Y") for i in df_petro['Date'] ]
datas_sem_pregao = [i for i in todas_datas if i not in datas_com_pregao]

df['title'] = df['title'].apply(lambda x: x.lower())
df['title'] = df['title'].apply(lambda x: "" if "petrobras" not in x else x)
df = df[(df['title'] != "")]

df_petro['Fechamento'] = df_petro['Var%'].apply(lambda x: 0 if x<0 else 1)

lista_datas = []
lista_news = []

for i in df.date.unique():
    news = ""
    for row in df[(df['date']==i)].iterrows():
        news = news + " " + row[1][0]
    lista_news.append(news)
    lista_datas.append(i)
    
    
df_news_diaria = pd.DataFrame(list(zip(lista_datas,lista_news)),
            columns =['Date', 'Noticias'])
df_news_diaria.sort_values(by = 'Date', ascending = True, inplace = True)

## Iterar sobre as datas dos pregões (iniciando pelo segundo dia do pregão de 2021 df_petro.Date.iloc[1:])

## Calcular delta (diferença entre dias entre dois registros seguidos de pregões):
import datetime

df_news_sem_pregao = pd.DataFrame()
timedelta_1dia = datetime.timedelta(days=1)

lista_datas = []
lista_noticias_sem_pregao = []

for i, data in enumerate(df_petro.Date.iloc[1:]):
    data_anterior = df_petro['Date'].iloc[i]  
    delta = data - data_anterior

    
    # Se houver mais de 1 dia sem pregão:    
    if delta > timedelta_1dia:
            
            
        # Filtra as noticias entre as datas sem pregão:
        df_aux = df_news_diaria[ (df_news_diaria['Date']> data_anterior) & (df_news_diaria['Date']<= data)  ]
        
        ## Concatena as noticias das datas sem pregão
        news = ""
        for row in df_aux.iterrows():
            news = news + " " + row[1][1]


        ## Armazena as noticias e data do ultimo pregão valido em listas
        lista_noticias_sem_pregao.append(news)
        lista_datas.append(data)
        
        #Cria um dataframe auxiliar com a data do ultimo pregão e as noticias concatenadas dos dias sem pregões:
        df_aux2 = pd.DataFrame(list(zip(lista_datas,lista_noticias_sem_pregao)),
            columns =['Date', 'Noticias'])
    
        # Gera o dataframe com as noticias sem pregões + datas do ultimo pregão valido.
        df_news_sem_pregao = df_news_sem_pregao.append(df_aux2, ignore_index = True)
        
        #Resetando as listas para geração de novo DF
        lista_noticias_sem_pregao = []
        lista_datas = []
        

df_news_diaria_atualizada = df_news_diaria.copy()

# itera sobre os dias com pregão cujo noticias de dias anteriores foram concatenadas:
for data in df_news_sem_pregao.Date.unique():
    
    #Filtra pelo dia com pregão que teve noticias concatenada
    df_noticia_dias_sem_pregao = df_news_sem_pregao[(df_news_sem_pregao['Date']==data)]


    #Checa se há registro referente a data no df de noticias
    df_check_noticias = df_news_diaria_atualizada[(df_news_diaria_atualizada['Date']==data)]
    
    # Se não houver registros referente á data então o registro deverá ser criado no df de noticias:
    # Se houver, então o registro será atualizado no df de noticias
    
    if len(df_check_noticias) > 0:
        
        #Substitui os registros
        df_news_diaria_atualizada = df_news_diaria_atualizada.replace ((df_news_diaria_atualizada.loc[df_news_diaria_atualizada['Date'].isin(df_noticia_dias_sem_pregao['Date'])])['Noticias'].values, df_noticia_dias_sem_pregao['Noticias'].values)
        
    else:
        #Insere o novo registro
        df_news_diaria_atualizada = df_news_diaria_atualizada.append(df_noticia_dias_sem_pregao, ignore_index = True)
        

        
df_final = pd.merge(left = df_petro, right = df_news_diaria_atualizada, how = 'left', on = 'Date')

df_final = df_final.dropna()
df_final = df_final[(df_final['Noticias'] != "")]

#### Gerando features Sentilex
sentilexpt = open('SentiLex-lem-PT01 editado v70_3.txt','r',encoding='utf-8-sig')

dic_palavra_polaridade = {}
for i in sentilexpt.readlines():
    pos_ponto = i.find('.')            # obtem a posiçãodo caracter ponto
    palavra = (i[:pos_ponto])          # Pega a palavra
    pol_pos = i.find('POL')            # obtem a posição do inicio da string POL
    polaridade = (i[pol_pos+4:pol_pos+6]).replace(';','')         # obtem a polaridade da palavra
    #polaridade = (i[pol_pos+4:pol_pos+7]).replace(';','')
    dic_palavra_polaridade[palavra] = polaridade                  # atualiza o dicionario com a palavra a polaridade
    

def Score_sentimento(frase):
    frase = frase.lower()                     # coloca toda a frase em minusculo
    l_sentimento = []                         # cria uma lista vazia
    for p in frase.split():
        l_sentimento.append(int(dic_palavra_polaridade.get(p, 0)))      # para cada palavra obtem a polaridade
        #l_sentimento.append(float(dic_palavra_polaridade.get(p, 0)))      # para cada palavra obtem a polaridade     
    #print (l_sentimento)                                                # imprime a lista de polaridades
    score = sum(l_sentimento)                                           # soma todos os valores da lista
    #if score > 0:
        #return 'Positivo, Score:{}'.format(score)                       # se maior que 0 retorna 'positivo'
    #elif score == 0:
        #return 'Neutro, Score:{}'.format(score)                         # se igual a 0 retorna 'neutro'
    #else:
        #return 'Negativo, Score:{}'.format(score)                       # se menor que 0 retorna 'negativo'
        
    return score

df_final2 = df_final.copy()

df_final2['score'] = df_final2['Noticias'].apply(lambda x: Score_sentimento(x))

### Traduzindo noticias
trans = Translator()
def traduzir(frase):
    frase = frase.lower()                     # coloca toda a frase em minusculo
    frase = trans.translate(frase, dest = 'en').text
    return frase

df_final3 = df_final.copy()
df_final3['Noticias'] = df_final3['Noticias'].apply(lambda x: traduzir(x))

#### Gerando features Roberta

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def neu_rob(frase):
    #trunca a frase para 514 caracteres (máximo suportado pelo modelo de Roberta)
    frase = frase[:514]
    
    encoded_text = tokenizer(frase, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    neg_roberta = scores[0]
# neu_roberta = scores[1]
# pos_roberta = scores[2]
    return neg_roberta

df_final4 = df_final.copy()
df_final4['Noticias'] = df_final3['Noticias']
df_final4['neu_rob'] = df_final4['Noticias'].apply(lambda x: neu_rob(str(x)))


#### Gerando features Finbert
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

df_bert = df_final.copy()
df_bert['Noticias'] = df_final3['Noticias']

def sentimento_finbert_neg(string):
    results = nlp([string])
    dict_results = results[0]
    sentimento = dict_results.get('label')
    
    if sentimento == "Negative":
        score = -1*dict_results.get('score')
    else:
        score = 0
        
    return score

df_bert['neg_finbert'] = df_bert['Noticias'].apply(lambda x: sentimento_finbert_neg(x))

# Sentilex + Vader
df_final8 = df_final3.merge(df_final2, how = 'left', on = ['Date', 'Adj Close', 'Volume', 'Var%', 'Fechamento'])
# + Roberta
df_final8 = df_final8.merge(df_final4, how = 'left', on = ['Date', 'Adj Close', 'Volume', 'Var%', 'Fechamento'])
# +Finbert
df_final8 = df_final8.merge(df_bert, how = 'left', on = ['Date', 'Adj Close', 'Volume', 'Var%', 'Fechamento'])

df_final99 = df_final8.copy()

features = ['score', 'neu_rob','neg_finbert']
featuresd1 = [i + "d1" for i in features]
featuresd2 = [i + "d2" for i in features]
featuresd3 = [i + "d3" for i in features]
featuresd4 = [i + "d4" for i in features]

#Criando as colunas de features para d-1, d-2, d-3, d-4 e inicializando com valores zeros:
for i in features:
    df_final99[i+"d1"] = 0
    df_final99[i+"d2"] = 0
    df_final99[i+"d3"] = 0
    df_final99[i+"d4"] = 0

#atualiza as features de d-1
for a,b in itertools.zip_longest(features,featuresd1):
    df_final99[b] = df_final99.shift(periods=1)[a]
    
#atualiza as features de d-2
for a,b in itertools.zip_longest(features,featuresd2):
    df_final99[b] = df_final99.shift(periods=2)[a]
    
#atualiza as features de d-3
for a,b in itertools.zip_longest(features,featuresd3):
    df_final99[b] = df_final99.shift(periods=3)[a]
    
#atualiza as features de d-4
for a,b in itertools.zip_longest(features,featuresd4):
    df_final99[b] = df_final99.shift(periods=4)[a]

df_final99 = df_final99.dropna()
df_final99.to_csv('df_final99.csv', index = False, sep='|')