from datetime import timedelta, date
import pandas as pd
from GoogleNews import GoogleNews
from datetime import datetime
import streamlit as st

def stack_minio(request):
    st.write("Inicio da Function")

    st.write("Definindo termo para busca da Function")
    termo = 'petrobras'
    st.write("Definindo termo " + termo + " para busca da Function")
    st.write("Definindo lang e region")
    googlenews = GoogleNews(lang='pt', region='BR')
    st.write("Definindo periodo de 23 horas para busca")
    googlenews = GoogleNews(period='12h')
    st.write("Fazendo chamada na API do Google News")
    googlenews.search(termo)
    st.write("Termino da chamada na API do Google News")
    result = googlenews.result()
    st.write("Fazendo o clear na API")
    googlenews.clear()
    st.write("Apresentando resultado")
    st.write(result)
    st.write("Termino da Function")


buscar = st.button('Buscar noticias')

if buscar:
    stack_minio(None)