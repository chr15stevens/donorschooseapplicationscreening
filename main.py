import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re, string

# a regex to replace all punctuation within a string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
# a function to run our regex and return our strings as arrays of words
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

# loading our training and test data and replacing missing values with blank strings to make concatenation easier
train = pd.read_csv('train_tmp.csv').replace(np.nan, '', regex=True)
test = pd.read_csv('test_tmp.csv').replace(np.nan, '', regex=True)

# concatenation of all descriptive columns into a single long column
train['project_description'] = train['project_title'].str.cat(train['project_essay_1']).str.cat(train['project_essay_2']).str.cat(train['project_essay_3']).str.cat(train['project_essay_4']).str.cat(train['project_resource_summary'])
test['project_description'] = test['project_title'].str.cat(test['project_essay_1']).str.cat(test['project_essay_2']).str.cat(test['project_essay_3']).str.cat(test['project_essay_4']).str.cat(test['project_resource_summary'])

print(train.head())

# We are using the TFIDF vectorizer to change our text into vectors of counts of words vs their appearances in documents
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
trainVector = vec.fit_transform(train['project_description'])
testVector = vec.transform(test['project_description'])

def pr(y_i, y):
    p = trainVector[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

# here we get the model we are building
def build_model(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = trainVector.multiply(r)
    return m.fit(x_nb, y), r

trainOutcomes = train['project_is_approved']
testOutcomes = test['project_is_approved']

m,r = build_model(trainOutcomes)
preds = m.predict_proba(testVector.multiply(r))[:,1] # check what this does

# here we compare the prediction probabilities from our test set to the actual values 'testOutcomes'
# we will use the mean squared error formula to compare our results

meanSquaredError = np.sum(np.square(testOutcomes.values - preds))/preds.size

print(meanSquaredError)
