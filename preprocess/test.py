import os
import pickle
import numpy

path = r"D:\math\NLP\project\Singing-Voice-Conversion-BingliangLi\preprocess\Opencpop\F0\train.pkl"

data = pickle.load(open(path, "rb"))
print(len(data))
max = 0
for i in data:
    if len(i) > max:
        max = len(i)
        
        
print(max) 