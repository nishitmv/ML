import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core.utils import gcs_utils
from bs4 import BeautifulSoup
import string

table = str.maketrans('', '', string.punctuation)
stopwords = ["A", "ABOUT", "ACTUALLY", "ALMOST", "ALSO", "ALTHOUGH", "ALWAYS", "AM", "AN", "AND", "ANY", "ARE", "AS", "AT", "BE", "BECAME", "BECOME", "BUT", "BY", "CAN", "COULD", "DID", "DO", "DOES", "EACH", "EITHER", "ELSE", "FOR", "FROM", "HAD", "HAS", "HAVE", "HENCE", "HOW", "I", "IF", "IN", "IS", "IT", "ITS", "JUST", "MAY", "MAYBE", "ME", "MIGHT", "MINE", "MUST", "MY", "MINE", "MUST", "MY", "NEITHER", "NOR", "NOT", "OF", "OH", "OK", "WHEN", "WHERE", "WHEREAS", "WHEREVER", "WHENEVER", "WHETHER", "WHICH", "WHILE", "WHO", "WHOM", "WHOEVER", "WHOSE", "WHY", "WILL", "WITH", "WITHIN", "WITHOUT", "WOULD", "YES", "YET", "YOU", "YOUR", "WAS"]
print("TABLE : ", table)

gcs_utils._is_gcs_disabled = True
trainingData = tfds.as_numpy(tfds.load('imdb_reviews', split='train', try_gcs=False))
testData = tfds.as_numpy(tfds.load('imdb_reviews', split='test', try_gcs=False))

imdb_sentences = []
labels = []
for item in trainingData:
    sentence = str(item['text'].decode('UTF-8').upper())
    soup = BeautifulSoup(sentence)
    sentence = soup.get_text()
    sentence = sentence.replace(",", " , ")
    sentence = sentence.replace(".", " . ")
    sentence = sentence.replace("-", " - ")
    sentence = sentence.replace("/", " / ")
    words = sentence.split()
    filteredSentence = ""
    for word in words:
        word = word.translate(table)
        if word not in stopwords:
            filteredSentence = filteredSentence+word+" "
    imdb_sentences.append(filteredSentence)
    labels.append(item['label'])
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=50000)
tokenizer.fit_on_texts(imdb_sentences)
sequences = tokenizer.texts_to_sequences(imdb_sentences)

test_imdb_sentences = []
test_labels = []
for item in testData:
    sentence = str(item['text'].decode('UTF-8').upper())
    soup = BeautifulSoup(sentence)
    sentence = soup.get_text()
    sentence = sentence.replace(",", " , ")
    sentence = sentence.replace(".", " . ")
    sentence = sentence.replace("-", " - ")
    sentence = sentence.replace("/", " / ")
    words = sentence.split()
    filteredSentence = ""
    for word in words:
        word = word.translate(table)
        if word not in stopwords:
            filteredSentence = filteredSentence+word+" "
    test_imdb_sentences.append(filteredSentence)
    test_labels.append(item['label'])



#print(tokenizer.word_index)
testSentences = [
'Today is a sunny day',
'Today is a rainy day',
'Is it sunny today?'
]

seqs = tokenizer.texts_to_sequences(testSentences)

print(seqs)
