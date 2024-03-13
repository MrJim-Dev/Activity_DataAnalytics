import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from wordcloud import WordCloud
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from lightgbm.sklearn import LGBMClassifier
import xgboost as xgb
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, 
                            recall_score, f1_score, roc_auc_score, classification_report, 
                            roc_curve, RocCurveDisplay)
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4') 

def lemmatization(words):
    lemm = WordNetLemmatizer()
    tokens = [lemm.lemmatize(word) for word in words]
    return tokens


st.set_page_config(layout="wide")
# Load dataset
df = pd.read_csv("datasets/spam.csv")


# Style the buttons to make them full-width
st.markdown("""
    <style>
        .stButton > button {
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# Style the buttons to make them full-width and add styling for custom metric container
st.markdown("""
    <style>
        .main {
            padding-left: 10%;
            padding-right: 10%;
        }
        .main > div {
            margin-top: -60px;
            text-align: center;
            display: flex;
            justify-content: center;
        }
        .stButton > button {
            width: 100%;
        }

        .customMetric {
            background-color: #f0f0f0;
            border-radius: 10px;
            padding: 20px;
            margin: 1%;
            font-size: 1.2em;
            text-align: center;
            width: 30%;  /* Adjusted to 50% width */
            float: left;  /* Make them float left */
        }

        .customMetric h2 {
            color: black;
            font-weight: bold;
            margin: 0;
            padding: 0;
        }

        .customMetric h5 {
            color: black;
        }

        .customMetric p {
            color: black;
        }

        .customMetric label {
            color: #888;
            font-size: 0.9em;
            display: block;
            margin-top: 10px;
        }

    </style>
""", unsafe_allow_html=True)

# Sidebar navigation using buttons


st.write("\n")


# Display content based on selected menu
st.header("Spam Email Prediction")
st.write("###### by James Alein Ocampo, Francis Michael Secuya")
st.markdown("""
The primary goal of this dashboard is to develop a predictive model that accurately classifies incoming email messages as either ham (non-spam) or spam. We will use the Email Spam Collection dataset, which consists of email messages tagged with their respective labels.

""")
st.write("\n")

st.markdown("<h2>Overview of the Categories</h2>", unsafe_allow_html=True)
st.write("\n")


# Total number of rows or records
total_records = len(df)

# Number of records with 'spam' category
spam_records = len(df[df['Category'] == 'spam'])

# Number of records with 'ham' category
ham_records = len(df[df['Category'] == 'ham'])


metrics_html = ""  # Collecting metrics HTML in this string

metrics_html += f"""
<div class="customMetric">
    <h5>Total Emails</h5>
    <h2>{total_records}</h2>
    <label>Total email messages analyzed</label>
</div>
"""

metrics_html += f"""
<div class="customMetric">
    <h5>Spam Emails</h5>
    <h2>{spam_records}</h2>
    <label>Spam emails identified</label>
</div>
"""
metrics_html += f"""
<div class="customMetric">
    <h5>Ham Emails</h5>
    <h2>{ham_records}</h2>
    <label>Ham emails identified</label>
</div>
"""


# Render the metrics
st.markdown(metrics_html, unsafe_allow_html=True)
st.markdown('<hr/>', unsafe_allow_html=True);

# Create two columns for the plots
col1, middle_space, col2 = st.columns([1, 0.1, 1])  # Adjust the middle space as needed (0.1 for 10% width)

# Plot the pie chart in the first column with some margin/padding
with col1:
    category_ct = df['Category'].value_counts().reset_index()
    category_ct.columns = ['Category', 'Count']

    # Create the pie chart
    fig = px.pie(category_ct, names='Category', values='Count', title='Pie Graph: spam or not')

    # Show the plot
    st.plotly_chart(fig)

# Add a little space in the middle
with middle_space:
    pass

# Plot the bar chart in the second column with some margin/padding
with col2:
    # Assuming df is your DataFrame
    categories = pd.get_dummies(df["Category"])
    spam_or_not = pd.concat([df, categories], axis=1)
    spam_or_not.drop('Category', axis=1, inplace=True)

    df["length"] = df["Message"].apply(len)

    ham = df.loc[np.where(spam_or_not['ham'] == 1)].reset_index()
    spam = df.loc[np.where(spam_or_not['ham'] == 0)].reset_index()

    ham.drop('index', axis=1, inplace=True)
    spam.drop('index', axis=1, inplace=True)

    ham_data = ham['length']
    spam_data = spam['length']

    # Creating histogram plot using Plotly
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=ham_data, name='Ham', marker_color='blue', opacity=0.75))
    fig.add_trace(go.Histogram(x=spam_data, name='Spam', marker_color='red', opacity=0.75))

    # The rest of the layout
    fig.update_layout(title_text='Length Distribution of Ham and Spam Messages', xaxis_title='Length', yaxis_title='Frequency', barmode='overlay')
    fig.update_traces(marker_line_color='black', marker_line_width=1.5)
    st.plotly_chart(fig)


#! Header title and description will be added here
# Render the metrics
st.markdown('<hr/>', unsafe_allow_html=True);

st.header("Ham and Spam Message Analysis")

st.write("This section shows the most used words in regular texts and spam, with charts and word clouds for each.\n")

col1, middle_space, col2 = st.columns([1, 0.1, 1])  # Adjust the middle space as needed (0.1 for 10% width)

# function to get all of strings from dataframe column, and used lower function here.
def get_all_str(df):
    sentence = ''
    for i in range(len(df)):
        sentence += df['Message'][i]
    sentence = sentence.lower()
    return sentence

def get_str(lst):
    sentence = ''
    for char in lst:
        sentence += char+' '
    sentence = sentence.lower()
    return sentence

# function to get words from text(string). used RegexpTokenizer
def get_word(text): 
    result = nltk.RegexpTokenizer(r'\w+').tokenize(text.lower())
#     result = result.lower()                                              
#     result = nltk.word_tokenize(text)
    return result

# function to add stopwords to nltp stopword list.
def stopword_list(stop):
    lst = stopwords.words('english')
    for stopword in stop:
        lst.append(stopword)
    return lst

# function to remove stopwords from list.
def remove_stopword(stopwords, lst):
    stoplist = stopword_list(stopwords)
    txt = ''
    for idx in range(len(lst)):
        txt += lst[idx]
        txt += '\n'
    cleanwordlist = [word for word in txt.split() if word not in stoplist] 
    return cleanwordlist

# function to get dataframe from cleanwordlist.
def Freq_df(cleanwordlist):
    Freq_dist_nltk = nltk.FreqDist(cleanwordlist)
    df_freq = pd.DataFrame.from_dict(Freq_dist_nltk, orient='index')
    df_freq.columns = ['Frequency']
    df_freq.index.name = 'Term'
    df_freq = df_freq.sort_values(by=['Frequency'],ascending=False)
    df_freq = df_freq.reset_index()
    return df_freq

# function to lemmatize words
def lemmatization(words):
    lemm = WordNetLemmatizer()
    tokens = [lemm.lemmatize(word) for word in words]
    return tokens

# function to plot word cloud of words
def Word_Cloud(data, color_background, colormap, title):
    # Convert word frequency dictionary to a DataFrame for Plotly
    df_freq = pd.DataFrame(list(data.items()), columns=['Word', 'Frequency'])
    df_freq = df_freq.sort_values('Frequency', ascending=False)

    # Create a bar chart using Plotly Express
    fig = px.bar(df_freq.head(50), # Display top 50 words for clarity
                x='Frequency', 
                y='Word', 
                orientation='h', 
                title=title)
    
    # Adjust layout for better readability
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, 
                    xaxis_title='Frequency', 
                    yaxis_title='Words')
    
    # Display the figure in Streamlit
    st.plotly_chart(fig)
import random

def generate_color_by_frequency(frequencies):
    max_freq = max(frequencies)
    colors = ['#' + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
            for freq in frequencies]
    return colors

def Word_Cloud_Plotly(data, color_background, title):
    sorted_data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True)[:100])
    
    words = list(sorted_data.keys())
    frequencies = list(sorted_data.values())
    colors = generate_color_by_frequency(frequencies)  # Generate colors for each word

    trace = go.Scatter(
        x=[random.random() for _ in words],  # Placeholder for actual positions
        y=[random.random() for _ in words],  # Placeholder for actual positions
        text=words,
        mode='text',
        textfont={
            'size': [2 + freq / max(frequencies) * 140 for freq in frequencies],  # Example size calculation
            'color': colors,  # Apply the generated colors to each word
        }
    )
    
    layout = go.Layout({
        'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
        'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
        'title': title,
    })
    
    fig = go.Figure(data=[trace], layout=layout)
    st.plotly_chart(fig)


nltk.download('stopwords')
string = get_all_str(ham)
words = get_word(string)
removed = remove_stopword('1',words)
# show 10 words for example
print(removed[:10])



# Plot the pie chart in the first column with some margin/padding
with col1:
    freq_df = Freq_df(removed) # Make sure this line correctly generates your freq_df
    top_10 = freq_df[:10]

    # Create a bar plot with Plotly
    fig = px.bar(top_10, x='Term', y='Frequency', 
                title='Rank of Ham Terms',
                labels={'Term': 'Term', 'Frequency': 'Frequency'},
                color='Frequency', # This line colors the bars by their frequency, using a default color scale
                text='Frequency') # This will add text labels with the frequency values

    # Customize the layout
    fig.update_layout(xaxis_tickangle=-45, # Rotate the labels to match matplotlib's appearance
                    xaxis_title='Term',
                    yaxis_title='Frequency',
                    xaxis=dict(tickmode='linear')) # Ensure the x-axis labels are linear and match the top_10 terms
    
    st.plotly_chart(fig)


# Add a little space in the middle
with middle_space:
    pass

# Plot the bar chart in the second column with some margin/padding
with col2:
    # Grouping by date and productivity, summing durations
    # Ensure 'Productive' is a categorical variable with 'Yes' first
    data = dict(zip(freq_df['Term'].tolist(), freq_df['Frequency'].tolist()))
    data = freq_df.set_index('Term').to_dict()['Frequency']
    ham_wordcloud = Word_Cloud_Plotly(data ,'white', 'Terms of Ham message')

# Plot the pie chart in the first column with some margin/padding
with col1:

    string = get_all_str(spam)
    words = get_word(string)
    removed = remove_stopword('1',words)

    freq_df = Freq_df(removed) # Make sure this line correctly generates your freq_df
    top_10 = freq_df[:10]

    # Create a bar plot with Plotly
    fig = px.bar(top_10, x='Term', y='Frequency', 
                title='Rank of Spam Terms',
                labels={'Term': 'Term', 'Frequency': 'Frequency'},
                color='Frequency', # This line colors the bars by their frequency, using a default color scale
                text='Frequency') # This will add text labels with the frequency values

    # Customize the layout
    fig.update_layout(xaxis_tickangle=-45, # Rotate the labels to match matplotlib's appearance
                    xaxis_title='Term',
                    yaxis_title='Frequency',
                    xaxis=dict(tickmode='linear')) # Ensure the x-axis labels are linear and match the top_10 terms
    
    st.plotly_chart(fig)

# Add a little space in the middle
with middle_space:
    pass

# Plot the bar chart in the second column with some margin/padding
with col2:
    data = dict(zip(freq_df['Term'].tolist(), freq_df['Frequency'].tolist()))
    data = freq_df.set_index('Term').to_dict()['Frequency']
    spam_wordcloud = Word_Cloud_Plotly(data ,'white', 'Terms of Spam message')


#! Header title and description will be added here
# Render the metrics
st.markdown('<hr/>', unsafe_allow_html=True);

st.header("Voting Classifiers")

st.write("This section compares the accuracy and AUC scores of various machine learning models in message classification and presents the cross-validation results of a Voting Classifier.\n")

col1, middle_space, col2 = st.columns([1, 0.1, 1])  # Adjust the middle space as needed (0.1 for 10% width)

def preprocess(sentence):
    words = get_word(sentence)
    words_ltz = lemmatization(words)
    removed = remove_stopword('1',words_ltz)
    return removed

df.replace('ham',1,inplace=True)
df.replace('spam',0,inplace=True)
df.head()

nltk.download('wordnet')

vector = CountVectorizer(analyzer = preprocess)
X = vector.fit(df['Message'])
X_transform = X.transform(df['Message'])

tfidf_transformer = TfidfTransformer().fit(X_transform)
X = tfidf_transformer.transform(X_transform)

train_X, test_X, train_y, test_y = train_test_split(X, df['Category'], test_size=0.30, random_state = 8888)    

rfc=RandomForestClassifier(random_state=8888)
lgbm = LGBMClassifier(boosting_type='gbdt',objective='binary',random_state=8888)
xgbr = xgb.XGBClassifier(objective='binary:hinge',random_state=8888)
svc = SVC(probability=True,random_state=8888)
catboost = CatBoostClassifier(random_state=8888, logging_level='Silent')

rfc.fit(train_X,train_y)
lgbm.fit(train_X, train_y)
xgbr.fit(train_X, train_y)
svc.fit(train_X, train_y)
catboost.fit(train_X,train_y,verbose=0)

classifiers = []
classifiers.append(svc)
classifiers.append(rfc)
classifiers.append(xgbr)
classifiers.append(lgbm)
classifiers.append(catboost)

model_name = ['SVC', 'Random Forest', 'XGBClassifier', 'LGBMClassifier', 'CatBoostClassifier']

accuracy_list = []
auc_list=[]
recall_list = []
f1_list = []

for classifier in classifiers :
    y_pred=classifier.predict(test_X)
    y_pred_proba=classifier.predict_proba(test_X)[:,1]
    accuracy_list.append(accuracy_score(test_y,y_pred))
    auc_list.append(roc_auc_score(test_y, y_pred_proba))
    recall_list.append(recall_score(test_y, y_pred))
    f1_list.append(f1_score(test_y, y_pred))

def plot_model_score(model_name, accuracy_list, auc_list, title):
    x = model_name  # Directly use model names for x-axis
    width = 0.2  # Adjust positions to simulate bar width

    # Creating two bar traces, one for accuracy and one for AUC
    bar1 = go.Bar(x=[x_val - width for x_val in range(len(x))], y=accuracy_list,
                name='Accuracy', opacity=0.7)

    bar2 = go.Bar(x=[x_val + width for x_val in range(len(x))], y=auc_list,
                name='AUC', opacity=0.7)

    # Combining the traces in a single figure
    fig = go.Figure(data=[bar1, bar2])

    # Adjusting layout for our figure
    fig.update_layout(title_text=title,
                    xaxis=dict(title='Models', tickmode='array', tickvals=list(range(len(x))), ticktext=model_name),
                    yaxis=dict(title='Score'),
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                    barmode='group',
                    bargap=0.15,  # Space between bars of adjacent location coordinates.
                    bargroupgap=0.1  # Space between bars of the same location coordinate.
                    )

    # Adding annotations for values on each bar
    for i, (acc, auc) in enumerate(zip(accuracy_list, auc_list)):
        fig.add_annotation(x=i - width, y=acc, text=f"{acc:.3f}", showarrow=False, yshift=10)
        fig.add_annotation(x=i + width, y=auc, text=f"{auc:.3f}", showarrow=False, yshift=10)

    # Displaying the plot in Streamlit
    st.plotly_chart(fig)

col1, middle_space, col2 = st.columns([1, 0.1, 1]) 

# Plot the pie chart in the first column with some margin/padding
with col1:
    title = "Model Performance Comparison"
    plot_model_score(model_name, accuracy_list, auc_list, title)


# Add a little space in the middle
with middle_space:
    pass

# Plot the bar chart in the second column with some margin/padding
with col2:
    kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=8888)
    votingC = VotingClassifier(estimators=[('light gbm', lgbm),('Random Forest', rfc),
                                        ('Cat boost',catboost)],voting='soft')

    votingC = votingC.fit(train_X, train_y)

    v_accuracy = cross_val_score(votingC, train_X, y = train_y, scoring = "accuracy", cv = kfold)
    v_auc = cross_val_score(votingC, train_X, y = train_y, scoring = "roc_auc", cv = kfold)

    votingC_accuracy_mean = v_accuracy.mean()
    votingC_auc_mean = v_auc.mean()

    metrics = ['Accuracy', 'AUC']
    mean_values = [votingC_accuracy_mean, votingC_auc_mean]

    # Create a bar chart
    fig = go.Figure(data=[
        go.Bar(x=metrics, y=mean_values, marker_color=['skyblue', 'lightcoral'])
    ])

    # Add annotations for each bar to display its value
    for i, value in enumerate(mean_values):
        fig.add_annotation(x=metrics[i], y=value,
                        text=str(round(value, 3)),
                        showarrow=False,
                        yshift=10)

    # Update the layout to add titles and labels
    fig.update_layout(
        title_text='Voting Classifier Cross-Validation Results',
        xaxis=dict(title='Metric'),
        yaxis=dict(title='Mean Score')
    )

    # Display the figure
    st.plotly_chart(fig)

st.write(f"Voting Classifier Accuracy: {votingC_accuracy_mean:.4f}")
st.write(f"Voting Classifier AUC: {votingC_auc_mean:.4f}")
    
