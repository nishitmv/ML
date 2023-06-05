import io
import json
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core.utils import gcs_utils
from bs4 import BeautifulSoup
import string
from keras_preprocessing.sequence import pad_sequences
import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt

table = str.maketrans('', '', string.punctuation)
stopwords = ["A", "ABOUT", "ACTUALLY", "ALMOST", "ALSO", "ALTHOUGH", "ALWAYS", "AM", "AN", "AND", "ANY", "ARE", "AS",
             "AT", "BE", "BECAME", "BECOME", "BUT", "BY", "CAN", "COULD", "DID", "DO", "DOES", "EACH", "EITHER", "ELSE",
             "FOR", "FROM", "HAD", "HAS", "HAVE", "HENCE", "HOW", "I", "IF", "IN", "IS", "IT", "ITS", "JUST", "MAY",
             "MAYBE", "ME", "MIGHT", "MINE", "MUST", "MY", "MINE", "MUST", "MY", "NEITHER", "NOR", "NOT", "OF", "OH",
             "OK", "WHEN", "WHERE", "WHEREAS", "WHEREVER", "WHENEVER", "WHETHER", "WHICH", "WHILE", "WHO", "WHOM",
             "WHOEVER", "WHOSE", "WHY", "WILL", "WITH", "WITHIN", "WITHOUT", "WOULD", "YES", "YET", "YOU", "YOUR",
             "WAS", "TO", "THE", "ON", "THIS", "U", "S", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
print("TABLE : ", table)

gcs_utils._is_gcs_disabled = True

with open('Sarcasm_Headlines_Dataset_v2.json') as f:
    datastore = json.load(f)

sentences = []
labels = []
urls = []

for item in datastore:
    sentence = item['headline'].upper()
    sentence = sentence.replace(",", " , ")
    sentence = sentence.replace(".", " . ")
    sentence = sentence.replace("-", " - ")
    sentence = sentence.replace("/", " / ")
    soup = BeautifulSoup(sentence)
    sentence = soup.get_text()
    words = sentence.split()
    filteredSentence = ""
    for word in words:
        word.translate(table)
        if word not in stopwords:
            filteredSentence = filteredSentence + word + " "
    sentences.append(filteredSentence)
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])
print(len(sentences))
training_size = 23000
training_sentences = sentences[0:training_size]
print(len(training_sentences))
test_sentences = sentences[training_size:]

training_labels = labels[0:training_size]
test_labels = labels[training_size:]

tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>", num_words=3000)
tokenizer.fit_on_texts(training_sentences)
trainingSeqs = tokenizer.texts_to_sequences(training_sentences)
trainingPaddedSeqs = pad_sequences(trainingSeqs, padding='post', maxlen=85)

wc = tokenizer.word_counts

newlist = (OrderedDict(sorted(wc.items(), key=lambda t: t[1], reverse=True)))
print(newlist)
xs = []
ys = []
curr_x = 1
for item in newlist:
    xs.append(curr_x)
    curr_x = curr_x + 1
    ys.append(newlist[item])
plt.axis([300, 3000, 0, 100])
plt.plot(xs, ys)
plt.show()

reverse_word_index = dict([(value, key)
                           for (key, value) in tokenizer.word_index.items()])

tokenizerTest = tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>", num_words=3000)
tokenizerTest.fit_on_texts(test_sentences)
testSeqs = tokenizerTest.texts_to_sequences(test_sentences)
testPaddedSeqs = pad_sequences(testSeqs, padding='post', maxlen=85)

trainingPaddedSeqs = np.array(trainingPaddedSeqs)
training_labels = np.array(training_labels)
testPaddedSeqs = np.array(testPaddedSeqs)
test_labels = np.array(test_labels)

print(len(trainingPaddedSeqs))

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(3000, 8),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(9, tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(1, tf.nn.sigmoid)
])
adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer=adam, loss=tf.losses.binary_crossentropy, metrics=['accuracy'])

model.summary()

model.fit(trainingPaddedSeqs, training_labels, epochs=80, validation_data=(testPaddedSeqs, test_labels))

sentences = ["granny starting to fear spiders in the garden might be real",
             "game of thrones season finale showing this sunday night",
             "TensorFlow book will be a best seller"]

evalSeq = tokenizer.texts_to_sequences(sentences)
print(evalSeq)

padded = pad_sequences(evalSeq, maxlen=85, padding='post')

print(padded)

print(model.predict(padded))

# model.evaluate(testPaddedSeqs, test_labels)
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, 3000):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()

