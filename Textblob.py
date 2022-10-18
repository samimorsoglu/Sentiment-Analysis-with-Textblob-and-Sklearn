# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 23:11:13 2022

@author: samim
"""

from textblob import TextBlob
import pandas as pd


metin = ["Beşiktaş şampiyon olacağına inanıyorum","Bu ödevi yapacağım.","Okuduğum bölümü seviyorum.","Bu ürün çok kötü."]

olumlu_yazilar=[]
olumsuz_yazilar=[]
olumlu_sonuc=[]
olumsuz_sonuc=[]
duygu_yok=[]

for yazi in metin:
    blob1 = TextBlob(yazi)
    blob_eng = blob1.translate(from_lang='tr', to='en')

    if(blob_eng.polarity > 0):
        olumlu_yazilar.append(yazi)
        olumlu_sonuc.append(blob_eng.sentiment)
    elif(blob_eng.polarity < 0):
        olumsuz_yazilar.append(yazi)
        olumsuz_sonuc.append(blob_eng.sentiment)
    else : 
        duygu_yok.append(yazi)
        
        
df=pd.DataFrame({"Olumlu Cümleler" : olumlu_yazilar,"Duygu Durumu":olumlu_sonuc })
df2=pd.DataFrame({"Olumsuz Cümleler" : olumsuz_yazilar,"Duygu Durumu":olumsuz_sonuc })
df3=pd.DataFrame({"Duygusu Olmayan Cümleler" : duygu_yok })
with pd.ExcelWriter('Turkish_Sentiment.xlsx') as writer:
    df.to_excel(writer, sheet_name='OLumlu')
    df2.to_excel(writer, sheet_name='Olumsuz')
    df3.to_excel(writer, sheet_name='Duygusuz')