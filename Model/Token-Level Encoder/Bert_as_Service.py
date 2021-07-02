import numpy as np 
from bert_serving.client import BertClient
bc = BertClient()

#Generate sentence representation by Bert-As-Service
text_all_embs = []
sentence_len = []
for i in range(len(text_all)):
    text = text_all[i].split("\n")
    sentence_len.append(len(text))
    text_embs = bc.encode(text)
    #text_embs = np.concatenate((text_embs,np.array([np.array(past_volatility_all[i])]*len(text_embs))),axis=1)
    text_all_embs.append(text_embs)
    
# Padding
dim = 1024 # Depends on the dimensions of your selected token-level pretrained model
b = np.zeros([len(text_all_embs),len(max(text_all_embs,key = lambda x: len(x))),dim]) 
for i,j in enumerate(text_all_embs): 
    b[i][0:len(j),:] = j 
