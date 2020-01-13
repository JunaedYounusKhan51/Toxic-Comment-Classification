from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
import csv
import pickle


texts = []
labels_toxic = []
labels_severe_toxic=[]
labels_obscene=[]
labels_threat=[]
labels_insult=[]
labels_identity_hate=[]
#label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

recipes = pd.read_csv('balanced_train_binary_data.csv')
X = np.array(recipes['comment_text'])
y_toxic = np.array(recipes['label'])




for i in range(len(X)):
    texts.append(unicode(X[i], errors='replace'))

for i in range(len(y_toxic)):
    labels_toxic.append(int(y_toxic[i]))


train_size = len(labels_toxic)


recipes = pd.read_csv('balanced_test_binary_data.csv')
X = np.array(recipes['comment_text'])
y_toxic = np.array(recipes['label'])




for i in range(len(X)):
    texts.append(unicode(X[i], errors='replace'))

for i in range(len(y_toxic)):
    labels_toxic.append(int(y_toxic[i]))


#print(texts)
'''
texts = [
    "good movies", "not a good movie", "did not like",
    "i like it", "good one"

]

print(texts)
labels = [
    "1","0","0","1","1"


]
'''
tfidf = TfidfVectorizer(min_df = 2, max_df = 0.5, ngram_range = (2,2), stop_words = 'english')
features = tfidf.fit_transform(texts)

print "ok"
#print features
'''
pd.DataFrame(
    features.todense(),
    columns=tfidf.get_feature_names()
)

#label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

print(tfidf.get_feature_names())
'''
#x_train, x_test, y_train, y_test = tts(features[7:], labels[7:], test_size=0.2)
x_train = features[0:train_size]
y_train_toxic = labels_toxic[0:train_size]


x_test = features[train_size:]
y_test_toxic = labels_toxic[train_size:]


#print len(y_train)
#print len(y_test)

#features_array = features.toarray()
###classifiers
clf_nb_toxic = MultinomialNB()


#clf_svm = svm.SVC(kernel='linear')
#clf_lr = LogisticRegression()

result_csv = []


#label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

###########################################################################

##predict and acc

print "toxic: "

clf_nb_toxic.fit(x_train, y_train_toxic)

pred = clf_nb_toxic.predict(x_test)
result_csv.append(pred)

print("------")

print ("test_accuracy: ")
print (accuracy_score(y_test_toxic, pred))

print ("test_precision: ")
print (precision_score(y_test_toxic, pred, average='weighted'))

print ("test_recall: ")
print (recall_score(y_test_toxic, pred, average='weighted'))

print ("test_f1 ")
print (f1_score(y_test_toxic, pred, average='weighted'))









######################














''' 

##model save code

##########################
filename = 'nb.sav'
loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X, y)
#print(result)



pred = loaded_model.predict(x_test)

print("###################")
print(".")
print("test results: ")
print("---------nb---------------------")
print ("test_accuracy: ")
print (accuracy_score(y_test, pred))

print ("test_precision: ")
print (precision_score(y_test, pred, average="weighted"))

print ("test_recall: ")
print (recall_score(y_test, pred, average="weighted"))

print ("test_f1 ")
print (f1_score(y_test, pred, average="weighted"))




filename = 'nb_pickle.pickle'
pickle.dump((y_test,pred), open(filename, 'wb'))

'''