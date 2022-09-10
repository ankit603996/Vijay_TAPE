from dash.exceptions import PreventUpdate
import dash
import dash_core_components as dcc
import dash_html_components as html
from io import BytesIO
import pandas as pd
import UDF
from dash import Dash, Input, Output, callback, dash_table
import base64
import io
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
#import nltk
#nltk.download('omw-1.4')
#nltk.download('stopwords')
from wordcloud import WordCloud,STOPWORDS
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]

stopwords = set(STOPWORDS)
def plot_wordcloud(df):
    comment_words = ''
    for val in df.Ticket_Summary:
        # typecaste each val to string
        val = str(val)
        # split the value
        tokens = val.split()
        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        comment_words += " ".join(tokens) + " "
    wordcloud = WordCloud(width=800, height=800,
                              background_color='white',
                              stopwords=stopwords,
                              min_font_size=5).generate(comment_words)
    return wordcloud.to_image()



def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('unicode_escape')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))

    except Exception as e:
        print(e)
        return None
    return df

from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize, pos_tag
#from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer


def frequency_count(df):
    class LemmaTokenizer(object):  # this lemmatization function will be used as an argument in countvectorizer
        def __init__(self):
            self.wnl = WordNetLemmatizer()
        def __call__(self, articles):
            return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]
    vectorizer = CountVectorizer(lowercase=True,ngram_range=(1,5), max_features=5000,
    stop_words='english', tokenizer=LemmaTokenizer())
    TFIDF = vectorizer.fit_transform(df['Ticket_Summary']).toarray()
    TFIDF = pd.DataFrame(TFIDF,columns =vectorizer.get_feature_names())
    TFIDF= TFIDF[[x for x in TFIDF.columns if len(x)>1]].T
    TFIDF['totalfrequency'] = TFIDF.sum(axis=1)
    TFIDF.sort_values("totalfrequency",inplace = True,ascending = False)
    TFIDF['tokens'] = TFIDF.index
    TFIDF = TFIDF[['tokens','totalfrequency']].reset_index(drop=True).iloc[:10,]
    return TFIDF

