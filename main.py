import tensorflow as tf
from CSVImporter import CSVImporter
from Modelbuilder import ModelBuilder

# initialize a tensor from the csv
importer = CSVImporter()
importer.read("8_seadoggs-vs-breakaway-dust2", 100)

# build a rnn model
model = ModelBuilder().buildModel()


# train the model
