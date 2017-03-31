# -*- coding: utf-8 -*-
import tweepy as tw
from sklearn.externals import joblib
from secrets import APIKEY,SECRET,ACESS_TOKEN,ACCESS_SECRET

from langdetect import detect
from our_utils  import stop_words, important_ngrams
import  snowballstemmer

stemmer = snowballstemmer.stemmer('english');


def to_ngrams(lista):
    salida = []
    for i in range(2,4):
        for index in range(0,len(lista)+1-i):
            salida += [lista[index:index+i]]
    return salida


with open('AdaBoost350.pkl', 'rb') as fid:
    model_loaded = cPickle.load(fid)
    

class MyStreamListener(tw.StreamListener):

    def on_status(self, status):
        # print("un estatus!")
        # print(status.text)
        # Detectamos si el tweet está en inglés
        try:
            lang = detect(status.text)
        except Exception as e:
            lang = "dunno"

        if (lang == 'en'):
            word_list  = status.text.replace(","," ").lower().split()
            
            no_sw_list = filter((lambda x:not x in stop_words),word_list)

            # stopword solo para 1-gram, stemming para todos
            stemmed_1    = stemmer.stemWords(no_sw_list)
            stemmed_2_3  = stemmer.stemWords(word_list)

            n_grams    = stemmed_1 + stemmed_2_3

            # Convirtiendo al vector compatible con nuestra tabla de datos
            word_vector = []
            for word in important_ngrams:
                word_vector += [int(word in n_grams)]
            print("important! "+ str(sum(word_vector))+ " words matched in table.")
            print(model_loaded.predict(word_vector))
            print(model_loaded.predict_proba(word_vector))

        else:
            # Lenguaje diferente al inglés
            print("not important (" + lang + ")")
            return (-1)

    # stop_words = get_stop_words('en')
    def on_error(self, status_code):
        if status_code == 420:
            #returning False in on_data disconnects the stream
            return False
        if status_code == 401:
            print("Error, intenta colocar la hora antigua de venezuela UTC -4:30")
        else:
            print("error %d" % status_code)


auth = tw.OAuthHandler(APIKEY, SECRET)
auth.set_access_token(ACESS_TOKEN, ACCESS_SECRET)

try:
    redirect_url = auth.get_authorization_url()
except tw.TweepError:
    print 'Error! Failed to get request token.'
    exit()

api = tw.API(auth)

myStream = tw.Stream(auth = api.auth, listener=MyStreamListener())
res = myStream.filter(track=['sad'])
print(res)

