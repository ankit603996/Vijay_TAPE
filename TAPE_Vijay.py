from dash.exceptions import PreventUpdate
import dash
import dash_core_components as dcc
import dash_html_components as html
from io import BytesIO
import pandas as pd
import UDF
from dash import Input, Output
import base64
import gensim
import gensim.corpora as corpora
import re

#df = pd.read_csv(r"F:\LocalDriveD\Analytics\Freelancing\Vijay\TAPE\Git\Sample.csv",encoding="unicode_escape")


# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__,suppress_callback_exceptions=True) #, external_stylesheets=external_stylesheets)


app.layout = html.Div([
    html.H2("Upload Excel/CSV File"),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False),
# dcc.Store stores the intermediate value
    dcc.Store(id='intermediate-value'),
    html.Div(id='dropdownUI'),
    html.Br(),
    html.Div(id ="wordcloudUI"),
    html.Br(),
    html.Div(id="topicmodelUI"),
    html.Br(),
    html.Div(id="wordfreqUI"),
])

# callback table creation
@app.callback(Output('intermediate-value', 'data'),
                Output('dropdownUI', 'children'),
              [Input('upload-data', 'contents'),
               Input('upload-data', 'filename')])
def update_output(contents, filename):
    if contents is not None:
        df = UDF.parse_contents(contents, filename)
        df.columns = [x.replace(".","_") for x in df.columns]
        df.columns = [x.replace(" ","_") for x in df.columns]
        return df.to_json(date_format='iso', orient='split'), \
               html.Div([
                   "Select column",
                   dcc.Dropdown(id="select_column",options=df.columns,value='')
               ])
    else:
        raise PreventUpdate

@app.callback(Output('wordcloudUI', 'children'),
                Output('topicmodelUI', 'children'),Output('wordfreqUI', 'children'),
              [Input('select_column', 'value'),
               Input('intermediate-value', 'data')])
def update_output(columnname, data):
    if (data is not None) & (columnname != ""):
        df = pd.read_json(data, orient='split')
        df[columnname] = df[columnname].astype(str)
        df[columnname] = [r.lower() for r in df[columnname]]
        # Punctuation and numeric remove
        df[columnname] = [re.sub(r'[?_><.,|/()!@#$%^&*:]|[-]|[\]|[0-9]', "", d, flags=re.I) for d in
                                df[columnname]]  # to include nueric user'[^ws]'
        # Strip white space
        df[columnname] = [re.sub(r"[\s]", " ", w, flags=re.I) for w in df[columnname]]
        df[columnname] = [r.lower() for r in df[columnname]]
        # Punctuation and numeric remove
        df[columnname] = [re.sub(r'[?_><.,|/()!@#$%^&*:]|[-]|[\]|[0-9]', "", d, flags=re.I) for d in
                                df[columnname]]  # to include nueric user'[^ws]'
        # Strip white space
        df[columnname] = [re.sub(r"[\s]", " ", w, flags=re.I) for w in df[columnname]]
        img = BytesIO()
        UDF.plot_wordcloud(df,columnname).save(img, format='PNG')
        src = 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())
        data = df[columnname].values.tolist()
        data_words = list(UDF.sent_to_words(data))
        # remove stop words
        data_words = UDF.remove_stopwords(data_words)
        # Create Dictionary
        id2word = corpora.Dictionary(data_words)
        # Create Corpus
        texts = data_words
        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]
        # number of topics
        num_topics = 10
        # Build LDA model
        try :
            lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                                   id2word=id2word,
                                                   num_topics=num_topics)

            topic_models = pd.DataFrame(lda_model.print_topics(), columns=["topic_number", "Words with Probability"])
            topic_models['topic_number'] = range(1, num_topics + 1)
            # word frequency count
            freq = UDF.frequency_count(df, columnname)
        except ValueError:
            topic_models = pd.DataFrame({'topic_number':0,
                                  "words with Probability":"No meaningful text found in data"},index=range(1))
            freq = pd.DataFrame({'words ':"No meaningful text found in data",
                                  "frequency":0},index=range(1))
        return UDF.wordcloud_ui("image_wc", src), \
               UDF.topicmodel_ui("datatable_lda", topic_models.to_dict('records'),[{"name": i, "id": i} for i in topic_models.columns]),\
               UDF.wordfreq_ui("datatable",  freq.to_dict('records'),[{"name": i, "id": i} for i in freq.columns])
    else:
        raise PreventUpdate

if __name__ == '__main__':
    app.run_server(debug=True)