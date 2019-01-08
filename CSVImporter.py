import tensorflow as tf
import pandas
import os
import json


class CSVImporter:


    def get_csv_as_tensor(self):
        file_name = "8_seadoggs-vs-breakaway-dust2"
        from pandas.io.json import json_normalize

        with open(file_name + ".json") as jsonFile:
            file = json.load(jsonFile)

        csv_file = json_normalize(file["GameInfo"], meta=["id"], record_path="players", )
        csv_file.to_csv(file_name + ".csv")

        # os.listdir()
        # dfiles = os.listdir('C:/Users/Juliu/PycharmProjects/git_LSTM_GRU')
        d_files = os.listdir()

        for file in d_files:
            print(file)

    df_json = pandas.read_json("8_seadoggs-vs-breakaway-dust2.json")
    df_csv = pandas.read_csv("8_seadoggs-vs-breakaway-dust2.csv")    #TODO: take in all (JSON/CSV )files ...

    # printing to console

    # print(dfjson)
    # this can be used to prints out the head of the table
    # (can be used to check what the file looks like...)
    print(df_json.head())
    print("\n")
    print(df_csv.head())


# return type should be tensor
# return
