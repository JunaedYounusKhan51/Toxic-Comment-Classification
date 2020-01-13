import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re

def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(dual=True) # prefer: dual = False, when n_sample > n_features
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r


train = pd.read_csv('balanced_train_data.csv')
test = pd.read_csv('balanced_test_data.csv')


label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1-train[label_cols].max(axis=1)
train.describe()



COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)

def tokenize(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text


n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
trn_term_doc = vec.fit_transform(train[COMMENT])
test_term_doc = vec.transform(test[COMMENT])



x = trn_term_doc
test_x = test_term_doc

preds = np.zeros((len(test), len(label_cols)))

for i, j in enumerate(label_cols):
    print('fit', j)
    m,r = get_mdl(train[j])
    preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]


#submid = pd.DataFrame({'id': subm["id"]})
#submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
pd.DataFrame(preds, columns = label_cols).to_csv('submission.csv', index=False)








from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import pandas as pd, numpy as np


test_labels = pd.read_csv('balanced_test_data.csv')

input_result_file = 'submission.csv'
submission = pd.read_csv(input_result_file)

submission = submission[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

threshold = .5
submission[submission >= threshold] = 1
submission[submission < threshold] = 0

submission.to_csv('check.csv', index=False)

categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

print "result of nb-logisitic::"
print "========================"

for category in categories:
    print('... Processing {}'.format(category))
    print('Test accuracy is {}'.format(accuracy_score(test_labels[category], submission[category])))
    print('Test precisin is {}'.format(precision_score(test_labels[category], submission[category],average='weighted')))
    print('Test recall is {}'.format(recall_score(test_labels[category], submission[category],average='weighted')))
    print('Test f1_score is {}'.format(f1_score(test_labels[category], submission[category],average='weighted')))