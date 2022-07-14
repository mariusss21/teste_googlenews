from datetime import timedelta, date
import pandas as pd
from GoogleNews import GoogleNews
from datetime import datetime
import streamlit as st
from time import sleep
from google.cloud import bigquery
from google.oauth2 import service_account
import plotly.graph_objects as go

st.set_page_config(
    page_title="Análise ações Petrobrás",
	layout="wide",
    #initial_sidebar_state="collapsed",
)

m = st.markdown("""
<style>
div.stButton > button:first-child{
    width: 100%;

}
</style>""", unsafe_allow_html=True)

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials)

@st.cache(show_spinner=True, ttl=3600)
def query_stackoverflow() -> pd.DataFrame:
    query_job = client.query("SELECT * FROM `marioloc-1491911271221.teste1.tabela1`")
    df = query_job.result().to_dataframe()  # Waits for job to complete.
    return df

        
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
    result = googlenews.result().to_dataframe()
    #st.write("Fazendo o clear na API")
    googlenews.clear()
    #st.write("Apresentando resultado")
    st.write(result)
    st.write("Termino da Function")
    





if __name__ == '__main__':

    st.subheader('Dados de ações da petrobrás')
    col1, col2 = st.columns([8, 2])

    st.sidebar.subheader('Stack Labs Finance')
    buscar = st.sidebar.button('Buscar noticias')
    select_bq = st.sidebar.button('bigquery test')

    if buscar:
        stack_minio(None)
            
    with col1:
        if select_bq:
            df = query_stackoverflow()
            st.write(df)
            fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'])])
            st.write(fig)


    
