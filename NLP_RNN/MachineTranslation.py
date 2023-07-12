import csv
import tensorflow_text as tftext
import pathlib
import tensorflow as tf
import numpy as np

def extractTSVData():
    hindata = []
    engdata = []
    with open("hin.txt") as f:
        csv_reader = csv.reader(f, delimiter="\t")
        for row in csv_reader:
            eng = row[0]
            hin = row[1]
            engdata.append(eng)
            hindata.append(hin)


def load_data(path: pathlib.Path):
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines()
    pairs = []
    for line in lines:
        pair = line.split('\t')
        pairs.append(pair)
   # print(pairs)
    context = np.array([context for target, context in pairs])
    target = np.array([target for target, context in pairs])
    print(context)
    print(target)
    return target, context

path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    extract=True)
path_to_file = pathlib.Path(path_to_zip).parent/'spa-eng/spa.txt'
target, context = load_data(path_to_file)