import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
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
               smooth_idf=1, sublinear_tf=1) # use max_features=1000 to restrict the number of words you want to consider
trainVector = vec.fit_transform(train['project_description'])
testVector = vec.transform(test['project_description'])

trainOutcomes = train['project_is_approved']
testOutcomes = test['project_is_approved']

# build our model
model = LogisticRegression(C=4, dual=True).fit(trainVector, trainOutcomes.values)
predictions = model.predict_proba(testVector)[:,1]

# here we compare the prediction probabilities from our test set to the actual values 'testOutcomes'
# we will use the mean squared error formula to compare our results

meanSquaredError = np.sum(np.square(testOutcomes.values - predictions))/predictions.size

print(meanSquaredError)
