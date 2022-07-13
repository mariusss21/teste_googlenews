from datetime import timedelta, date
import pandas as pd
from GoogleNews import GoogleNews
from datetime import datetime, date
import streamlit as st
from time import sleep
from google.cloud import bigquery
from google.oauth2 import service_account
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import yfinance as yf
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import stylecloud
from stop_words import get_stop_words

st.set_page_config(
    page_title="Análise ações Petrobrás",
	layout="wide",
    initial_sidebar_state="expanded",
)

m = st.markdown("""
<style>
div.stButton > button:first-child{
    width: 100%;
}
div.block-container{
    padding-top: 1rem;
}
</style>""", unsafe_allow_html=True)
    

@st.cache(show_spinner=True, ttl=3600)
def raw_petro() -> pd.DataFrame:
    df = pd.read_csv('raw_petro.csv')
    df.Date = pd.to_datetime(df.Date).dt.date
    return df


@st.cache(show_spinner=True, ttl=3600)
def raw_gnews() -> pd.DataFrame:
    df = pd.read_csv('raw_gnews.csv', index_col=0)
    df.date = pd.to_datetime(df.date).dt.date
    return df


@st.cache(show_spinner=True, ttl=3600)
def final_df() -> pd.DataFrame:
    df = pd.read_csv('df_final.csv', sep='|')
    df.Date = pd.to_datetime(df.Date).dt.date
    return df


def petro_chart(df_raw_petro_date):
    fig = go.Figure(data=[go.Candlestick(x=df_raw_petro_date['Date'],
                open=df_raw_petro_date['Open'],
                high=df_raw_petro_date['High'],
                low=df_raw_petro_date['Low'],
                close=df_raw_petro_date['Close'])])

    fig.update_layout(
		height=300,
		margin=dict(b=5,	t=0,	l=0,	r=0),
        font=dict(size=15),
        xaxis_rangeslider_visible=False)

    st.plotly_chart(fig, use_container_width=True)


def latest_news(df):
    df_noticias_view = df.tail(10)
    st.subheader('Últimas notícias')
    with st.expander('Últimas 10 notícias'):
        for i in range(df_noticias_view.shape[0]):
            row = df_noticias_view.iloc[i]
            st.markdown("---")
            st.markdown(f"""
                ##### {i+1}. {row['media']} - **{row['title']}**
                {row['desc']}\n
                *{row['date']}* 
                """)


def qtd_news(df: pd.DataFrame, df_raw_petro: pd.DataFrame):
    df_count = df.groupby('date').count().copy()
    df_count.reset_index(inplace=True)
    df_count.rename(columns={'date': 'Date'}, inplace=True)
    df_chart = df_raw_petro.merge(df_count, on='Date', how='left')

    fig = go.Figure(data=[go.Bar(x=df_raw_petro.Date,
                        y=df_chart.title,
                        #marker={'color': label}
                        )])
    
    fig.update_layout(
		height=100,
		margin=dict(b=5,	t=0,	l=0,	r=0),
        font=dict(size=15),
        xaxis_rangeslider_visible=False,
        xaxis_visible=False)
        #title_text='Quantidade de notícias coletadas por dia')
    
    fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.6)

    st.plotly_chart(fig, use_container_width=True)


def live_values(df_petr4: pd.DataFrame, df_ibov: pd.DataFrame, dia: str):
    st.subheader(f'Cotação {dia}')

    st.metric(label="PETR4",
     value=round(df_petr4.tail(1)['Adj Close'].item(), 2),
     delta=round((df_petr4.tail(1)['Adj Close'].item() - df_petr4.head(1)['Open'].item()) * 100 /df_petr4.head(1)['Open'].item(), 2),
     delta_color="normal")

    st.metric(label="Índice Bovespa",
     value=round(df_ibov.tail(1)['Adj Close'].item(), 2), 
     delta=round(float(df_ibov.tail(1)['Adj Close'].item() - df_ibov.head(1)['Open'].item())* 100 /df_ibov.head(1)['Open'].item(), 2), 
     delta_color="normal")

    st.subheader('Previsão para o dia')


def word_cloud(df_news):
    # Especificar a coluna de titulo do DataFrame
    #summary = df_news['title']
    titulos = " ".join(s for s in df_news['title'])
    descricao = " ".join(s for s in df_news['desc'])
    all_summary = titulos + " " + descricao

    # Concatenar as palavras e remover de String
    #all_summary = " ".join(s for s in summary)
    all_summary = all_summary.replace("\n", "")
    all_summary = all_summary.replace(".", "")
    all_summary = all_summary.replace(",", "")
    all_summary = all_summary.replace("?", "")
    all_summary = all_summary.replace(",", "")
    all_summary = all_summary.replace("|", "")
    all_summary = all_summary.replace("(", " ")
    all_summary = all_summary.replace(")", "")
    all_summary = all_summary.replace("Petrobras", "Petrobrás")
    all_summary = all_summary.replace(" ", "  ")

    # Lista de stopword
    stopwords = set(STOPWORDS)
    stopwords.update(["da", "meu", "em", "você", "de", "ao", "os", "mês", "ano", "neste", "podem", "pelo", 'e', 'é', 'que', 'se', 'o', 'a', 'um', 'uma', 'para', 'na', 'pela', 'por', 'à'])

    # Gerar uma wordcloud
    wordcloud = WordCloud(stopwords=stopwords,
                          background_color="black",
                          width=400, height=250).generate(all_summary)

    # Mostrar a imagem final
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_axis_off()

    # Carrega Imagem
    wordcloud.to_file("sumario_wordcloud.png")
    st.image("sumario_wordcloud.png")


def news_sources(df):
    df['title'] = df['title'].apply(lambda x: "" if "petrobras" not in x else x)
    dfmed1 = df.copy()
    dfmed1['media'] = dfmed1['media'].str.replace('Click Petróleo e Gás','CPG Click Petroleo e Gas')
    dfmed = dfmed1.groupby(['media']).count()
    dfmed.sort_values(by='title', ascending=False, inplace=True)
    dfmed.reset_index(inplace=True)
    totnot = dfmed['title'].sum()
    dfmed['title'] = pd.to_numeric(dfmed['title'])
    dfmed['perc'] = ((dfmed['title'] / totnot) *100)
    dfmed['perc'] = dfmed['perc'].round(2)
    dfmed.sort_values(by='perc', ascending=False, inplace=True)
    dfmed.reset_index(inplace=True)
    dfmed = dfmed.head(30)
    dfmed.sort_values(by='perc', ascending=True, inplace=True)

    fig = go.Figure(data=[go.Bar(y=dfmed.media, x=dfmed.perc, orientation='h')])
    fig.update_layout(
        height=475,
        margin=dict(b=5,	t=0,	l=0,	r=0),
        font=dict(size=15),
        )
    
    fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.6)

    st.plotly_chart(fig, use_container_width=True)


def dashboard(data_inicial, data_final):
    #coletando os dados
    df_raw_petro = raw_petro()
    df_raw_petro_date = df_raw_petro.loc[(df_raw_petro['Date'] >= data_inicial) & (df_raw_petro['Date'] <= data_final)]

    df_raw_gnews = raw_gnews()
    df_raw_gnews_date = df_raw_gnews.loc[(df_raw_gnews['date'] >= data_inicial) & (df_raw_gnews['date'] <= data_final)]
    
    df_final = final_df()
    df_final_date = df_final.loc[(df_final['Date'] >= data_inicial) & (df_final['Date'] <= data_final)]
    
    st.title('Dashboard Petrobrás')
    col1, col2 = st.columns([8, 2])

    tipo_cotacao = st.sidebar.radio('Cotação', ['Histórica', 'Dia'])
    if datetime.now().hour >= 13 and datetime.now().minute > 5:
        date_ = date.today()
        texto = 'Atual'
    else: 
        date_ = date.today() - timedelta(days=1)
        texto = 'de ontem'

    df_petr4 = yf.download('PETR4.SA', start=date_, interval = "1m")
    df_petr4.reset_index(inplace=True)
    df_petr4.rename(columns={'Datetime': 'Date'}, inplace=True)

    df_ibov = yf.download('^BVSP', start=date_, interval = "1m")
    df_ibov.reset_index(inplace=True)
    df_ibov.rename(columns={'Datetime': 'Date'}, inplace=True)

    with col1:
        if tipo_cotacao == 'Histórica':
            st.subheader('Cotação histórica')
            petro_chart(df_raw_petro_date)
        if tipo_cotacao == 'Dia':
            st.subheader('Cotação dia')
            petro_chart(df_petr4)

        qtd_news(df_raw_gnews_date, df_raw_petro_date)
        latest_news(df_raw_gnews)

    with col2:
        live_values(df_petr4, df_ibov, texto)



if __name__ == '__main__':

    st.sidebar.image("logo_stack.png", use_column_width=True)
    st.sidebar.title('Equipe MinIO')
    st.sidebar.write('StackLabs finanças')
    col1, col2 = st.columns([8, 2])

    paginas = ['Análise ações Petrobrás', 'Notícias', 'Equipe MinIO']
    pagina = st.sidebar.radio('Menu', paginas)

    data_inicial = st.sidebar.date_input('Data inicial', datetime.now() - timedelta(days=30))
    data_final = st.sidebar.date_input('Data final', datetime.now() - timedelta(days=1))
    
    if pagina == 'Análise ações Petrobrás':
        dashboard(data_inicial, data_final)

    if pagina == 'Notícias':
        st.title('Análise de notícias')
        col1, col2 = st.columns(2)
        df_raw_gnews = raw_gnews()
        df_raw_gnews_date = df_raw_gnews.loc[(df_raw_gnews['date'] >= data_inicial) & (df_raw_gnews['date'] <= data_final)]
        with col1:
            st.subheader('Fontes das notícias')
            news_sources(df_raw_gnews_date)
        
        with col2:
            st.subheader('Palavras mais frequentes')
            word_cloud(df_raw_gnews)

        st.subheader('Notícias')

        with st.expander('Notícias do período'):
            news_qtd = st.number_input('Quantidade de notícias', value=10, min_value=1, max_value=1000)
            df_noticias_view = df_raw_gnews_date.sort_values('date', ascending=False)
            df_noticias_view = df_noticias_view.head(news_qtd)
            st.write(df_noticias_view)
            for i in range(df_noticias_view.shape[0]):
                row = df_noticias_view.iloc[i]
                st.markdown("---")
                st.markdown(f"""
                    ##### {i+1}. {row['media']} - **{row['title']}**
                    {row['desc']}\n
                    *{row['date']}* 
                    """)
            
    # with col1:
    #     if select_bq:
    #         df = query_stackoverflow()
    #         st.write(df)
    #         fig = go.Figure(data=[go.Candlestick(x=df['Date'],
    #                     open=df['Open'],
    #                     high=df['High'],
    #                     low=df['Low'],
    #                     close=df['Close'])])
    #         st.write(fig)


    
