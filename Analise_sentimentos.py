#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json


# In[ ]:


credenciais = {}
credenciais ['CONSUMER_KEY'] = 'P4cHsAWV4xxuY3bpdZz7GhSoT' 
credenciais ['CONSUMER_SECRET'] = 'XTJ6eSgQkjYsKZe8BrkvWljTHrVHbOIXGrGfgIQI9HIGCNZjC7' 

#salvar as credenciais num arquivo

with open ('C:/Users/junio/Downloads/analise_sentimentos/twitter_credenciais.json', 'w') as file:
  json.dump(credenciais, file)


# In[ ]:


pip install twython


# In[ ]:


#Importando Twython
from twython    import Twython

#Importando biblioteca Pandas
import pandas as pd


#Carregando credenciais do arquivo json
with open('C:/Users/junio/Downloads/analise_sentimentos/twitter_credenciais.json', 'r') as file:
  creds = json.load(file)


#Instanciando objeto
python_tweets   = Twython(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])

#Declarando dicionário
dict_ =  {'id':[], 'user': [], 'date' : [], 'text' : [], 'favorite_count': [], 'retweet_count': [], 'location':[]}

def buscar_tweets(query_term, max_id, max_iters):

  for call in range(0, max_iters):
    #Configurar a query
    query = { 'q'            : query_term,
          'result_type'      : 'recent',
          'count'            : 100,
          'lang'             : 'pt',
          'max_id'           : max_id,
          'tweet_mode'       : 'extended',
          'include_entities' : False
    }

    for status in python_tweets.search(**query)['statuses']:
      if 'RT @' not in (status['full_text']):
        dict_['id'].append(status['id'])
        dict_['user'].append(status['user']['screen_name'])
        dict_['date'].append(status['created_at'])
        dict_['text'].append(status['full_text'])
        dict_['favorite_count'].append(status['favorite_count'])
        dict_['retweet_count'].append(status['retweet_count'])
        dict_['location'].append(status['user']['location'])
        max_id = status['id']


# In[ ]:


import random
import time


# In[ ]:


termos      = ['rio de janeiro', 'claudio castro', 'seplag rio']
print('Serão extraídos tweets sobre os ' + str(len(termos)) + ' termos.')

for termo in termos:
  print('Iniciando buscas de ' + termo)

  buscar_tweets(termo, "", 50)
  df = pd.DataFrame(dict_)
  df.to_csv('C:/Users/junio/Downloads/analise_sentimentos/Todos_termos.csv', index=False)

  print('Final do processamento de ' + termo)
  randomico   = random.randrange(10, 30)
  print('Vamos aguardar ' + str(randomico) +' segundos.')   
  time.sleep(randomico) 
  
df


# In[ ]:


df.location.value_counts()


# In[ ]:


df  = pd.read_csv('C:/Users/junio/Downloads/analise_sentimentos/Todos_termos.csv')


# In[ ]:


df


# In[ ]:


df  = df.drop_duplicates(['text'])


# In[ ]:


df.count()


# In[ ]:


df = df.reset_index()


# In[ ]:


df.drop(columns=['index', 'id', 'user', 'date', 'favorite_count', 'retweet_count', 'location'], inplace=True)


# In[ ]:


df


# In[ ]:


df.to_csv('C:/Users/junio/Downloads/analise_sentimentos/Termos_tratados.csv')


# In[2]:


import pandas as pd


# In[ ]:


df = pd.read_csv('C:/Users/junio/Downloads/analise_sentimentos/Termos_tratados.csv')


# In[ ]:


pip install googletrans==4.0.0rc1


# In[ ]:


from googletrans import Translator


# In[ ]:


translator = Translator()

translate_text = translator.translate("tudo bem?", src='pt')
print(translate_text);


# In[ ]:


translator = Translator()

translate_text = translator.translate("tudo bem?", src='pt')
print(translate_text.text);


# In[ ]:


df


# In[ ]:


import time
from time import sleep
import random


# In[ ]:


suffix  = pd.DataFrame({'suffix': ['com', 'com.br', 'com.ar', 'pt', 'gr', 'ca']})

tamanho = len(df)
i       = 0

#Criando o df_t para receber os tweets traduzidos
df_t    = pd.DataFrame()

while i<tamanho:
    if(i>0 and i%150 == 0):
      aleatorio_centena   = random.randrange(3000, 9000)
      sleep(aleatorio_centena/1000)
      print("Passou pelo registro de número ", i)

    suffix_aleatorio    = random.randrange(0,5)

    
    traduzido   = translator.translate(df.text[i], src='pt')

    df_t = pd.concat([df_t, pd.DataFrame({'translated': [traduzido.text]})], ignore_index = True)

    aleatorio_granel   =  random.randrange(300, 3500)
    sleep(aleatorio_granel/1000)
    df_t.to_csv('C:/Users/junio/Downloads/analise_sentimentos/Termos_tratados_traduzidos.csv', index = False)
    i+=1

df_t


# In[ ]:


tamanho = len(df)
i       = 0

#Criando o df_t para receber os tweets traduzidos
df_t    = pd.DataFrame()

while i<tamanho:
    if(i>0 and i%150 == 0):
      aleatorio_centena   = random.randrange(10, 30)
      sleep(aleatorio_centena/15)
      print("Passou pelo registro de número ", i)

     
    traduzido   = translator.translate(df.text[i], src='pt')

    df_t = pd.concat([df_t, pd.DataFrame({'translated': [traduzido.text]})], ignore_index = True)

    aleatorio_granel   =  random.randrange(10, 15)
    sleep(aleatorio_granel/10)
    df_t.to_csv('C:/Users/junio/Downloads/analise_sentimentos/Termos_tratados_traduzidos.csv', index = False)
    i+=1

df_t


# In[ ]:


df_t


# In[3]:


df   = pd.read_csv('C:/Users/junio/Downloads/analise_sentimentos/Termos_tratados.csv')
df_t = pd.read_csv('C:/Users/junio/Downloads/analise_sentimentos/Termos_tratados_traduzidos.csv')


# In[4]:


df['translated'] = df_t


# In[5]:


df.drop(columns=['Unnamed: 0'], inplace=True)


# In[6]:


df


# In[7]:


text  = df['text']


# In[8]:


text


# In[9]:


uma_string  = " ".join(s for s in text)


# In[10]:


uma_string


# In[ ]:


get_ipython().system('pip install wordcloud -q')


# In[11]:


import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[12]:


stopwords = set(STOPWORDS)
stopwords.update(['da', 'meu', 'em', 'você', 'seu', 'de', 'ao', 'os', 'que', 'minha', 'do'])


# In[13]:


stopwords


# In[14]:


wordcloud = WordCloud(stopwords=stopwords,
                      background_color='black',
                      width=1600,
                      height=800).generate(uma_string)


# In[15]:


fig, ax = plt.subplots(figsize=(16,8))

ax.imshow(wordcloud, interpolation='bilinear')
ax.set_axis_off()

plt.imshow(wordcloud)
wordcloud.to_file('nuvem.png')


# In[26]:


pip install spacy
pip install nltk


# In[29]:


get_ipython().system('pip install spacy')
get_ipython().system('python -m spacy download pt_core_news_sm')


# In[30]:


import string
import re
import spacy
import nltk


# In[31]:


lista_stop_nuvem  = nltk.corpus.stopwords.words('portuguese')
pln_nuvem         = spacy.load('pt_core_news_sm')
pontuacao_nuvem   = string.punctuation


# In[32]:


lista_stop_nuvem


# In[33]:


def processar_nuvem(texto_nuvem):
  texto_nuvem         = texto_nuvem.lower()
  find                = r'(@[A-Za-z0-9áéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ$@-_.&+]+)|(https?://[A-Za-z0-9./]+)'
  replace             = r' '
  texto_nuvem         = re.sub(find, replace, texto_nuvem)
  texto_nuvem         = re.sub(r" +", " ", texto_nuvem)

  tokens_nuvem        = pln_nuvem(texto_nuvem)
  lista_palavras_nuvem= []
  for token_nuvem in tokens_nuvem:
    lista_palavras_nuvem.append(token_nuvem.text)

  lista_palavras_nuvem    = [token for token in lista_palavras_nuvem if token not in pontuacao_nuvem and token not in lista_stop_nuvem]
  lista_palavras_nuvem    = " ".join(lista_palavras_nuvem)

  return lista_palavras_nuvem


# In[34]:


processar_nuvem("Testando a, retirada de @folha_sp http://www.com.br aqui.")


# In[35]:


text2 = df['text'].apply(processar_nuvem)


# In[36]:


text2


# In[37]:


uma_string2   = " ".join(s for s in text2)


# In[38]:


uma_string2


# In[39]:


wordcloud = WordCloud(background_color='black',
                      width=1600,
                      height=800).generate(uma_string2)


# In[40]:


fig, ax = plt.subplots(figsize=(16,8))

ax.imshow(wordcloud, interpolation='bilinear')
ax.set_axis_off()

plt.imshow(wordcloud)
wordcloud.to_file('nuvem2.png')


# In[41]:


text3 = df['text'].apply(processar_nuvem)


# In[42]:


text3


# In[43]:


uma_string3   = " ".join(s for s in text3)


# In[44]:


uma_string3


# In[45]:


wordcloud = WordCloud(background_color='black',
                      width=1600,
                      height=800).generate(uma_string3)


# In[46]:


fig, ax = plt.subplots(figsize=(16,8))

ax.imshow(wordcloud, interpolation='bilinear')
ax.set_axis_off()

plt.imshow(wordcloud)
wordcloud.to_file('nuvem3.png')


# In[55]:


def retirar_termos(texto_retirar):
  texto_retirar     = re.sub(r"claudio|castro|rio|janeiro|cláudio|ca", " ", texto_retirar)
  return texto_retirar


# In[56]:


uma_string4   = retirar_termos(uma_string3)


# In[57]:


wordcloud = WordCloud(background_color='black',
                      width=1600,
                      height=800).generate(uma_string4)


# In[58]:


fig, ax = plt.subplots(figsize=(16,8))

ax.imshow(wordcloud, interpolation='bilinear')
ax.set_axis_off()

plt.imshow(wordcloud)
wordcloud.to_file('nuvem4.png')


# In[ ]:




