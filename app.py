from datetime import timedelta, date
import pandas as pd
import numpy as np
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
import pickle
import joblib
from dateutil import parser

st.set_page_config(
    page_title="Análise ações Petrobrás",
	layout="wide",
    initial_sidebar_state="expanded",
)


credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials)


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
    # df = pd.read_csv('raw_petro.csv')
    # df.Date = pd.to_datetime(df.Date).dt.date

    query_job = client.query("SELECT * FROM `stack-minio.dts_stack_minio.raw_yfinance`")
    df = query_job.result().to_dataframe()
    df.Date = pd.to_datetime(df.Date).dt.date
    df.drop_duplicates(inplace=True)
    return df


@st.cache(show_spinner=True, ttl=3600)
def raw_gnews() -> pd.DataFrame:
    # df = pd.read_csv('raw_gnews.csv', index_col=0)
    # df.date = pd.to_datetime(df.date).dt.date

    query_job = client.query("SELECT * FROM `stack-minio.dts_stack_minio.raw_googlenews`")
    df = query_job.result().to_dataframe()
    df.date = pd.to_datetime(df.date, format="%d/%m/%Y").dt.date
    df.drop_duplicates(inplace=True)
    return df


@st.cache(show_spinner=True, ttl=3600)
def final_df() -> pd.DataFrame:
    df = pd.read_csv('df_final.csv', sep='|')
    df.Date = pd.to_datetime(df.Date).dt.date
    return df


def petro_chart(df_raw_petro_date: pd.DataFrame):
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


def latest_news(df: pd.DataFrame):
    df_noticias_view = df.sort_values(by='date', ascending=False).head(10)
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
    #st.write(df.loc[df['date'] == date(2022, 6, 2)])
    df_count = df.copy()
    df_count['contador'] = 1
    df_count = df_count[['date', 'contador']].groupby('date').sum()
    df_count.reset_index(inplace=True)
    df_count.rename(columns={'date': 'Date'}, inplace=True)
    df_chart = df_raw_petro.merge(df_count, on='Date', how='left')

    fig = go.Figure(data=[go.Bar(x=df_raw_petro.Date,
                        y=df_chart.contador,
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


def live_values(df_petr4: pd.DataFrame, df_ibov: pd.DataFrame, df_news: pd.DataFrame, dia: str):
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
    
    st.write('Aqui vai ficar a previsão')
    today = datetime.now() - timedelta(hours=3)
    quantidade = df_news.loc[df_news['date'] == today.date()].shape[0]
    st.write(f'Quantidade de notícias hoje: {quantidade}')


@st.cache(show_spinner=True, ttl=3600)
def predict_model(df: pd.DataFrame):
    ensemblevote = joblib.load(open('model_ensemble.pkl', 'rb'))
  
    df_final99 = df.copy()
    df_final99['Date'] = pd.to_datetime(df_final99['Date'])

    treinar_ate_data_str = '2022-02-02'
    data_limite_teste = '2022-07-20'

    X_train = df_final99[(df_final99['Date'] <= parser.parse(treinar_ate_data_str))][['score','neu_robd4','neg_finbertd2','scored3']]
    y_train = df_final99[(df_final99['Date'] <= parser.parse(treinar_ate_data_str))][['Fechamento']]
    y_train = np.ravel(y_train)   
        
    X_test = df_final99[(df_final99['Date'] > parser.parse(treinar_ate_data_str)) & (df_final99['Date'] <= parser.parse(data_limite_teste))][['score','neu_robd4','neg_finbertd2','scored3']]
    X_test2 = df_final99[(df_final99['Date'] > parser.parse(treinar_ate_data_str)) & (df_final99['Date'] <= parser.parse(data_limite_teste))][['Date', 'score','neu_robd4','neg_finbertd2','scored3']]
    y_test = df_final99[(df_final99['Date'] > parser.parse(treinar_ate_data_str)) & (df_final99['Date'] <= parser.parse(data_limite_teste))][['Fechamento']]
    y_test = np.ravel(y_test) 

    ensemblevote.fit(X_train,y_train)

    #Predizendo y
    y_pred = ensemblevote.predict(X_test)
    X_test2['y_pred'] = y_pred
    ensemblevote.score(X_test, y_test)
    return X_test2


def model_chart(df: pd.DataFrame, df_raw_petro: pd.DataFrame):
    df_aux = df.copy()
    df_aux['Date'] = pd.to_datetime(df_aux['Date']).dt.date
    df_chart = df_raw_petro.merge(df_aux[['Date', 'y_pred']], on='Date', how='left')
    df_chart.loc[df_chart['y_pred'] == 0] = -1
    df_chart['y_pred'] = df_chart['y_pred'] * 5
    df_chart['y_pred'].fillna(0, inplace=True)

    #df_chart['colors'] = 'green' if df_chart['y_pred'] == 1 else 'red'
    df_chart.loc[df_chart['y_pred'] == 5, 'colors'] = 'green'
    df_chart.loc[df_chart['y_pred'] == -5, 'colors'] = 'red'
    list_colors = list(df_chart['colors'])

    fig = go.Figure(data=[go.Bar(x=df_raw_petro.Date,
                        y=df_chart.y_pred,
                        marker={'color': list_colors}
                        )])
    
    fig.update_layout(
		height=100,
		margin=dict(b=5,	t=0,	l=0,	r=0),
        font=dict(size=15),
        xaxis_rangeslider_visible=False,
        xaxis_visible=False)
        #title_text='Quantidade de notícias coletadas por dia')
    
    #fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
    #              marker_line_width=1.5, opacity=0.6)
    st.plotly_chart(fig, use_container_width=True)


def word_cloud(df_news: pd.DataFrame):
    # Especificar a coluna de titulo do DataFrame
    #summary = df_news['title']
    titulos = " ".join(s for s in df_news['title']) #as
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
                           height=260).generate(all_summary)

    # Mostrar a imagem final
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_axis_off()

    # Carrega Imagem
    wordcloud.to_file("sumario_wordcloud.png")
    st.image("sumario_wordcloud.png",use_column_width=True)


def news_sources(df: pd.DataFrame):
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
    if datetime.now().hour >= 14 and datetime.now().minute > 15:
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
        
        df_predict = predict_model(df_final)
        model_chart(df_predict, df_raw_petro_date)
        qtd_news(df_raw_gnews_date, df_raw_petro_date)
        latest_news(df_raw_gnews)

    with col2:
        live_values(df_petr4, df_ibov, df_raw_gnews, texto)

    with col1:
        predict_model(df_final)


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
            word_cloud(df_raw_gnews_date)

        st.subheader('Notícias')

        with st.expander('Notícias do período'):
            news_qtd = st.number_input('Quantidade de notícias', value=10, min_value=1, max_value=1000)
            df_noticias_view = df_raw_gnews.loc[(df_raw_gnews['date'] >= data_inicial) & (df_raw_gnews['date'] <= data_final)]
            df_noticias_view.sort_values(by='date', ascending=False, inplace=True)
            df_noticias_view = df_noticias_view.head(int(news_qtd))
            for i in range(df_noticias_view.shape[0]):
                row = df_noticias_view.iloc[i]
                st.markdown("---")
                st.markdown(f"""
                    ##### {i+1}. {row['media']} - **{row['title']}**
                    {row['desc']}\n
                    *{row['date']}* 
                    """)

    
