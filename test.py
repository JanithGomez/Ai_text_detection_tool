import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import anderson_ksamp
from math import log10
import os
import re
from pypdf  import PdfReader
from collections import Counter
from transformers import logging
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Set the page layout to wide
st.set_page_config(layout="wide")


# Initialize sentiment analysis pipeline
logging.set_verbosity_error()
absolute_path = os.path.dirname(__file__)
relative_path = "./roberta-base-openai-detector/"
full_path = os.path.join(absolute_path, relative_path)
tokenizer = AutoTokenizer.from_pretrained(full_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(full_path, local_files_only=True)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Function to preprocess text: remove numerics and special characters
def preprocess_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    return cleaned_text

# Function to calculate word length distribution
def word_length_distribution(text):
    # Preprocess text
    text = preprocess_text(text)
    # Tokenize text into words
    words = text.split()
    # Calculate word lengths
    word_lengths = [len(word) for word in words]
    # Create a Pandas Series for word length distribution
    word_length_counts = pd.Series(word_lengths).value_counts().sort_index()
    return word_length_counts

# Function to calculate sentence length distribution
def sentence_length_distribution(text):
    # Split text into sentences
    sentences = text.split('.')
    # Calculate sentence lengths
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    # Create a Pandas Series for sentence length distribution
    sentence_length_counts = pd.Series(sentence_lengths).value_counts().sort_index()
    return sentence_length_counts

# Function to calculate vocabulary richness
def calculate_vocabulary_richness(text):
    # Preprocess text
    text = preprocess_text(text)
    # Tokenize text into words
    words = text.split()
    # Calculate vocabulary richness
    unique_words = set(words)
    vocabulary_richness = len(unique_words) / len(words)
    return vocabulary_richness

# Function to calculate punctuation usage
def calculate_punctuation_usage(text):

    # Define the main punctuation marks to include
    main_punctuation_marks = r'[.,!?;:()]'

    # Count punctuation marks
    punctuation_counts = Counter(re.findall(main_punctuation_marks, text))
    return punctuation_counts

def calculate_first_letter_distribution(paragraphs):
    letter_counts = {}
    for paragraph in paragraphs:
        cleaned_paragraph = preprocess_text(paragraph)
        words = cleaned_paragraph.split()
        first_letters = [word[0].lower() for word in words if word]
        for letter in first_letters:
            letter_counts[letter] = letter_counts.get(letter, 0) + 1
    
    total_letters = sum(letter_counts.values())
    letter_proportions = {letter: count / total_letters for letter, count in letter_counts.items()}
    letter_proportions = {letter: proportion if not pd.isna(proportion) else 0.0 for letter, proportion in letter_proportions.items()}
    
    first_letter_distribution = pd.Series(letter_proportions)
    
    return first_letter_distribution

# Function to read text from uploaded file
def read_uploaded_file(uploaded_file):
    text = ""
    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            text = uploaded_file.getvalue().decode("utf-8")
            text = text.split('\n\n') 
        elif uploaded_file.type == "application/pdf":
            text = read_text_from_pdf(uploaded_file)
    return text

# Function to read text from a PDF file
def read_text_from_pdf(uploaded_file):
    
    reader = PdfReader(uploaded_file)
    
    # Initialize an empty list to store paragraphs
    paragraphs = []
    
    # Iterate through each page in the PDF
    for page_num in range(len(reader.pages)):

        if page_num >= 2:
            # Extract text from the page
            page = reader.pages[page_num]
            text = page.extract_text()
            
            # Split text into paragraphs
            paragraphs.extend(text.split('\n\n'))  # Adjust delimiter based on PDF formatting

    return paragraphs


# Display the image
st.image('dashboard.png', caption='Benford Law Analysis Dashboard', use_column_width=True)

# Upload a file
uploaded_file = st.file_uploader("Upload a PDF or text file", type=["pdf", "txt"])

# Create a Streamlit app
st.title('Model Analysis')

# Add a horizontal line
st.markdown("<hr>", unsafe_allow_html=True) 

def convert_pdf_to_text(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
      page = pdf_reader.pages[page_num]
      text += page.extract_text()
    return text

# Check if a file is uploaded
if uploaded_file:
    # Read the text from the uploaded file
    paragraphs = read_uploaded_file(uploaded_file)
    text = convert_pdf_to_text(uploaded_file)
    if text:
        res = classifier(text, truncation=True, max_length=510)
        label = res[0]['label']
        score = res[0]['score']

        if label == 'Real':
            real_score = score * 100
            fake_score = 100 - real_score
        else:
            fake_score = score * 100
            real_score = 100 - fake_score

        st.markdown(f"**Human Written Score: {real_score:.2f}%**")
        st.markdown(f"**Ai Written Score: {fake_score:.2f}%**")

    # Create a Streamlit app
    st.title('Benford Law Analysis')

    # Add a horizontal line
    st.markdown("<hr>", unsafe_allow_html=True) 

    # print(paragraphs)
    # Check if there are paragraphs extracted
    if paragraphs:
        # Create a DataFrame from the paragraphs
        df = pd.DataFrame({'Paragraphs': paragraphs, 'Index': range(len(paragraphs))})

        # Calculate first-letter distribution for each paragraph and store in a list
        first_letter_distributions = [calculate_first_letter_distribution(paragraph) for paragraph in df['Paragraphs']]
        aggregate_distribution = pd.concat(first_letter_distributions).groupby(level=0).mean()

        # Extract letters and their corresponding proportions
        letters = [index for dist in first_letter_distributions for index in dist.index]
        proportions = [proportion for dist in first_letter_distributions for proportion in dist]

        # Sort the data in descending order of proportions
        sorted_data = sorted(zip(letters, proportions), key=lambda x: x[1], reverse=True)
        letters_sorted, proportions_sorted = zip(*sorted_data)

        # Perform Anderson-Darling test
        # Benford's Law expected frequencies for first digits (1 to 9)
        benford_expected_flat = [np.log10(1 + 1/d) for d in range(1, 10)]
        ad_statistic, critical_values, significance_levels = anderson_ksamp([proportions_sorted[:9], benford_expected_flat])

        st.set_option('deprecation.showPyplotGlobalUse', False)
        
        # Print the Anderson-Darling test results
        # Print the Anderson-Darling test results
        st.write("Anderson-Darling statistic:", ad_statistic)

        # Display critical values in a table
        critical_values_table = pd.DataFrame({'Critical Value': critical_values}, index=[f"Critical value {i + 1}" for i in range(len(critical_values))])
        st.write("Critical values:")
        st.table(critical_values_table)

        # Display significance levels
        st.write("Significance levels:", significance_levels)

        # Interpret results
        if ad_statistic < critical_values[1]:
            st.write("**The observed distribution significantly deviates from Benford's Law. Hence confirms Ai Generated.**")
        else:
            st.write("**The observed distribution confirms to Benford's Law. Hence confirms Human Written.**")

        # Add a horizontal line
        st.markdown("<hr>", unsafe_allow_html=True) 

        # Create a Streamlit app
        st.title('Stylometric Analysis')

        try:
            text = uploaded_file.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            # If decoding with UTF-8 fails, try decoding with 'latin1'
            text = uploaded_file.getvalue().decode("latin1")

        # Calculate and display corpus size
        corpus_size = len(text)
        st.write(f"Corpus size: {corpus_size} characters")

        # Calculate and display word length distribution
        word_length_counts = word_length_distribution(text)
        st.write("Word length distribution:")
        st.bar_chart(word_length_counts)

        # Calculate and display sentence length distribution
        sentence_length_counts = sentence_length_distribution(text)
        st.write("Sentence length distribution:")
        st.bar_chart(sentence_length_counts)

        # Calculate and display vocabulary richness
        vocabulary_richness = calculate_vocabulary_richness(text)
        st.write(f"Vocabulary richness: {vocabulary_richness}")

        # Calculate and display punctuation usage
        punctuation_counts = calculate_punctuation_usage(text)
        st.write("Punctuation usage:")
        st.bar_chart(punctuation_counts)
    else:
        st.write("No paragraphs found in the uploaded file.")
