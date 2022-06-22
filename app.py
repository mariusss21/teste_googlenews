from datetime import timedelta, date
import pandas as pd
from GoogleNews import GoogleNews
from datetime import datetime
import streamlit as st
from time import sleep
from google.cloud import bigquery
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials)

def query_stackoverflow():

    query_job = client.query("SELECT * FROM 'marioloc-1491911271221.teste1.tabela1' LIMIT 10")

    results = query_job.result()  # Waits for job to complete.

    for row in results:
        print("{} : {} views".format(row.url, row.view_count))
        
        
def stack_minio(request):
    st.write("Inicio da Function")

    #st.write("Definindo termo para busca da Function")
    termo = 'petrobras'
    #st.write("Definindo termo " + termo + " para busca da Function")
    #st.write("Definindo lang e region")
    googlenews = GoogleNews(lang='pt', region='BR', period='12h')
    #st.write("Definindo periodo de 23 horas para busca")
    #googlenews = GoogleNews(period='12h')
    #st.write("Fazendo chamada na API do Google News")
    googlenews.search(termo)
    #st.write("Termino da chamada na API do Google News")
    result = googlenews.result()
    #st.write("Fazendo o clear na API")
    googlenews.clear()
    #st.write("Apresentando resultado")
    st.write(result)
    st.write("Termino da Function")

    

buscar = st.button('Buscar noticias')

if buscar:
    for i in list(range(1,200)):
        st.write(i)
        stack_minio(None)
        sleep(2)
        
select_bq = st.button('bigquery test')

if select_bq:
    query_stackoverflow()



    
