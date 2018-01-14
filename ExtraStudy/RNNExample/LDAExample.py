from nltk.tokenize import RegexpTokenizer#자연어 처리를 위한 라이브러리
from stop_words import get_stop_words#필요없는 문자를 지우기 위한 라아비르리
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models#LDA모델을 만들기 위한 라이브러리
import gensim

doc_a = "Broccolli is good to eat. My brother likes to eat good broccolli, but not my mother."
doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
doc_e = "Health professionals say that brocolli is good for your health."

# compile sample documents into a list
doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]

for doc in doc_set:
    #각 단어마다 문서를 자른다.
    tokenizer = RegexpTokenizer(r'\w+')
    raw = doc.lower()
    tokens = tokenizer.tokenize(raw)
    print(tokens)
    en_stop = get_stop_words('en')

    #필요없는 문자들을 걸러낸다.
    stopped_tokens = [i for i in tokens if i not in en_stop]
    print(stopped_tokens)

    #단어를 기본적인 형태로 변형
    p_stemmer = PorterStemmer()
    texts = [p_stemmer.stem(i) for i in stopped_tokens]

    print(texts)

    dictionary = corpora.Dictionary([texts])
    corpus = [dictionary.doc2bow(texts)]

    print(dictionary)

    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word = dictionary, passes=20)
    print(ldamodel.print_topics(num_topics=2, num_words=4))
    print("------------------------")



