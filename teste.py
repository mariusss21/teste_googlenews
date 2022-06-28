

def stack_minio(request):
    print ("Inicio da Function")
    from datetime import timedelta, date
    import pandas as pd
    from GoogleNews import GoogleNews
    from datetime import datetime
    print ("Definindo termo para busca da Function")
    termo = 'petrobras'
    print ("Definindo termo " + termo + " para busca da Function")
    print ("Definindo lang e region")
    googlenews = GoogleNews(lang='pt', region='BR')
    print ("Definindo periodo de 23 horas para busca")
    googlenews = GoogleNews(period='12h')
    print ("Fazendo chamada na API do Google News")
    googlenews.search(termo)
    print ("Termino da chamada na API do Google News")
    result = googlenews.result()
    print ("Fazendo o clear na API")
    googlenews.clear()
    print ("Apresentando resultado")
    print (result)
    print ("Termino da Function")

stack_minio(None)