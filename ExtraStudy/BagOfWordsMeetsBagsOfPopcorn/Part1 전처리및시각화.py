import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from multiprocessing import Pool
import numpy as np

#Pandas를 이용하여 csv 읽기 header는 column의 위치 delimiter는 각 필드를 파싱 quoting은 큰따음표를 무시하는 역활을 한다.
train = pd.read_csv('data/labeledTrainData.tsv', header=0, delimiter='\t', quoting=3)
test = pd.read_csv('data/testData.tsv', header=0, delimiter="\t", quoting=3)
#
# from bs4 import BeautifulSoup
#
# example1 = BeautifulSoup(train['review'][0], "lxml")
#
# import re
#
# letters_only = re.sub('[^a-zA-Z]', " ", example1.getText())
#
#
# lower_case = letters_only.lower()
# words = lower_case.split()
#
# from nltk.corpus import stopwords
#
# words = [w for w in words if not w in stopwords.words('english')]
#
# from nltk.stem import SnowballStemmer
#
# stemmer = SnowballStemmer('english')
# words = [stemmer.stem(w) for w in words]
#
# print(words)

stemmer = WordNetLemmatizer()

def review_to_words( raw_review ):
    review_text = BeautifulSoup(raw_review, 'html.parser').getText()
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words('english'))
    meaningful_words = [w for w in words if not w in stops]
    stemming_words = [stemmer.lemmatize(w) for w in meaningful_words]

    return(' '.join(stemming_words))

# train['review_clean'] = train['review'].apply(review_to_words)

def _apply_df(args):
    df, func, kwargs = args
    return df.apply(func, **kwargs)

def _apply_by_multiprocessing(df, func, **kwargs):
    workers = kwargs.pop('workers')
    pool = Pool(processes=workers)
    result = pool.map(_apply_df , [(d, func, kwargs) for d in np.array_split(df, workers)])
    pool.close()
    return pd.concat(list(result))

time_clean_train_reviews = _apply_by_multiprocessing(train['review'], review_to_words, workers=4)
print(time_clean_train_reviews)
