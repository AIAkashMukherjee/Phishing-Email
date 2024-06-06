import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from imblearn.over_sampling import SMOTE
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

 # type: ignore


def load_dataset(path):
    return pd.read_csv(path)

def data_overview(df):
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\nSummary statistics:")
    print(df.describe())
    print("\nData types and missing values:")
    print(df.info())

def preprocess(text):
    stop_words = stopwords.words("english")
    tokens=word_tokenize(text)

    tokens=[token.lower() for token in tokens]

    tokens=[re.sub(r'[^\w\s]','',token)for token in tokens]

    tokens=[re.sub(r'#\S+','',token)for token in tokens]
    tokens=[re.sub(r'@\S+','',token)for token in tokens]

    stop_words=set(stopwords.words('english'))
    tokens=[token for token in tokens if token not in stop_words]
    
    text=' '.join(tokens)
    return text


def clean_data(df):
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df['date']= pd.to_datetime(df['date'], format='%a, %d %b %Y %H:%M:%S %z',errors='coerce',utc=True)
    df['Date'] = df['date'].dt.date
    df['Time'] = df['date'].dt.time
    df.dropna(subset=['date'],inplace=True)
    df['body']=df['body'].apply(preprocess)
    # print(df)
    return df


def prepare_data(df, text_column, target_column):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf_vectorizer.fit_transform(df[text_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42), tfidf_vectorizer


# Train model
def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"Accuracy: {acc}")
    # return acc

# Save model and vectorizer
def save_model(model, vectorizer, model_filename, vectorizer_filename):
    joblib.dump(model, model_filename)
    joblib.dump(vectorizer, vectorizer_filename)


# Main function
def main(filepath, text_column, target_column, model_filename, vectorizer_filename):
    df = load_dataset(filepath)
    data_overview(df)
    df = clean_data(df)
    (X_train, X_test, y_train, y_test), tfidf_vectorizer = prepare_data(df, text_column, target_column)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, tfidf_vectorizer, model_filename, vectorizer_filename)


# Execute main function
if __name__ == "__main__":
    try:
        data_filepath = 'data/CEAS_08.csv'  # Replace with your actual file path
        text_column_name = 'body'  # Replace with your actual text column name
        target_column_name = 'label'  # Replace with your actual target column name
        model_output_filename = 'LogisticRegression_model.pkl'
        vectorizer_output_filename = 'tfidf_vectorizer.pkl'
        main(data_filepath, text_column_name, target_column_name, model_output_filename, vectorizer_output_filename)
    except Exception as e:
        print(e)
   


