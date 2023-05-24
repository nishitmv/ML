import urllib.request
import zipfile

url = "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"
test_url = "https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip"
filename = "../horse-or-human.zip"
testFileName = "validation-horse-or-human.zip"
training_dir = '../horse-or-human/training'
test_dir = '../horse-or-human/test'
urllib.request.urlretrieve(url, filename)

zip_refer = zipfile.ZipFile(filename, 'r')
zip_refer.extractall(training_dir)
zip_refer.close()

urllib.request.urlretrieve(test_url, testFileName)

zip_refer_test = zipfile.ZipFile(testFileName, 'r')
zip_refer_test.extractall(test_dir)
zip_refer_test.close()

