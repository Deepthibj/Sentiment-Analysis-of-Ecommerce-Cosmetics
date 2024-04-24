from langdetect import detect
import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Function to clean text data
def clean_text(text):
    # Remove emojis, convert to lowercase, remove non-numeric characters and punctuations
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    return text.strip()

# Streamlit App
st.title("Sentiment Prediction App")

# Upload file through Streamlit
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Read the data into a Pandas DataFrame
        if uploaded_file.name.endswith('csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')

        # Check if DataFrame has more than one column
        if len(df.columns) > 1:
            st.error("Error: Uploaded file should contain only one review column.")
        else:
            # Convert DataFrame to string
            df.dropna(inplace=True)
            df_str = df.to_string(index=False, header=False)
         
            # Data Cleaning
            st.header("Data Cleaning:")

            cleaned_rows = []
            for row in df_str.split('\n'):
                cleaned_text = clean_text(row)
                if cleaned_text:
                    cleaned_rows.append([cleaned_text])

            # Create DataFrame from cleaned rows
            df_cleaned = pd.DataFrame(cleaned_rows)

            # Display the cleaned data
            st.write("Cleaned Data:")
            st.dataframe(df_cleaned)

            if df_cleaned.empty:
                st.warning("No textual data found for prediction.")
            else:
                # Load the TF-IDF vectorizer from pickle
                with open('tfidf_vectorizer_final.pkl', 'rb') as file:
                    tfidf_vectorizer = pickle.load(file)

                # Vectorize text data
                X = tfidf_vectorizer.transform(df_cleaned.iloc[:, 0])  

                # Make predictions
                st.header("Predictions:")

                # Load the pre-trained machine learning model
                with open('ovr_classifier_model_final.pkl', 'rb') as file:
                    loaded_model = pickle.load(file)

                # Make predictions
                predictions = loaded_model.predict(X)

                # Add predictions to DataFrame
                df_cleaned['Predicted_Sentiment'] = predictions

                st.write(df_cleaned)

                # Plot sentiment distribution
                st.subheader("Sentiment Distribution:")
                sentiment_counts = df_cleaned['Predicted_Sentiment'].value_counts()
                st.bar_chart(sentiment_counts)

    except Exception as e:
        st.error(f"An error occurred: {e}")
