import pandas
import json


class CSVImporter:

    def __init__(self):
        self.batches = []

    def read(self, file_name, batch_size):

        from pandas.io.json import json_normalize

        with open(file_name + ".json") as jsonFile:
            file = json.load(jsonFile)

        csv_file = json_normalize(file["GameInfo"], meta=["id"], record_path="players")
        csv_file.to_csv(file_name + ".csv")

        df = pandas.read_csv(file_name + '.csv', usecols=[1,2,3,5,6,7,8,9])

        alldata = []
        #can be optimized because it iterates over the code 10 times instead of just putting all the data to the right batch
        #directly
        for x in range(0, 10):
            for i, r in df.iterrows():
                if i%10 == x:
                    alldata.append(r.values)

            self.batches.append([])

            for i in range(0, (len(alldata)//batch_size) -1):
                self.batches[0].append(alldata[i*batch_size:i*batch_size+batch_size-1])

            alldata = []










        #print(csv_file)




