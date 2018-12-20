import tensorflow as tf
import pandas
import os

#dfiles = os.listdir('C:/Users/Juliu/PycharmProjects/git_textGenRNN_GRU_LSTM')
dfiles = os.listdir()

for file in dfiles:
   print (file)

#os.listdir()

#class CSVImporter:

    # return type should be tensor
   # def getCSVasTensor(self):
dfjson = pandas.read_json("8_seadoggs-vs-breakaway-dust2.json")
print(dfjson)
    #    return

