#!/usr/bin/env python
# coding: utf-8

# In[11]:


#安裝隨https://medium.com/@br19920702/%E4%B8%AD%E7%A0%94%E9%99%A2%E9%96%8B%E6%BA%90nlp%E5%A5%97%E4%BB%B6-ckiptagger-%E7%B9%81%E4%B8%AD%E4%B8%8D%E7%B5%90%E5%B7%B4-7822fc4efbf
from ckiptagger import WS, POS, NER
import pandas as pd
import time
import os
from multiprocessing import Pool
import multiprocessing as mp
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

df1 = pd.read_csv('s104_textCombined_0727.csv')
df1 = df1.head(3)

df1.insert(3,'WS','')
df1.insert(4,'POS','')
df1.insert(5,'NER','')

word_list = [x for x in range(len(df1))]

#讀入斷詞數據包
#ws = WS('data')
#pos = POS("./data")
#ner = NER("./data")

ws = WS('data', disable_cuda=False)
pos = POS("./data", disable_cuda=False)
ner = NER("./data", disable_cuda=False)


#定義測試funtion_這邊是改版給單句用的
def word_seg(sentence_list): #輸入包含多斷文章(句子)的list
    word_sentence_list = ws([sentence_list],
                sentence_segmentation=True,
                segment_delimiter_set={'?', '？', '!', '！', '。', ',','，', ';', ':','：','\\n','\\r','.'}
                # recommend_dictionary = dictionary1, # words in this dictionary are encouraged
                # coerce_dictionary = dictionary2, # words in this dictionary are forced
               )
    pos_sentence_list = pos(word_sentence_list)

    entity_sentence_list = ner(word_sentence_list, pos_sentence_list)
    
    return word_sentence_list[0], pos_sentence_list, entity_sentence_list


def main(i):    
    word_processed = word_seg(df1['Combined'][i])
    df1.loc[i, 'WS'] = word_processed[0]
    df1.loc[i, 'POS'] = word_processed[1]
    df1.loc[i, 'NER'] = word_processed[2]

if __name__ == '__main__':
    t1 = time.time()
    pool = Pool() # Pool() 不放參數則默認使用電腦核的數量
    pool.map(main,word_list) 
    pool.close()
    pool.join()
    print('Total time: %.1f s' % (time.time()-t1))  


    df1.to_csv('s104_textCombined_ws_pos_ner_0801.csv')