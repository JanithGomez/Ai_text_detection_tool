# Ai_text_detection_tool
# AI Text Detection Using Benford's Law Integration 

This application performs Benford's Law analysis and Stylometric analysis on text data uploaded by the user. It provides insights into whether the text is likely to be human-written or AI-generated based on statistical analysis and linguistic patterns, Also it further confirms if the given texts in the file using an open sourced model.

## Model Information
The application utilizes a text detecion model based on the RoBERTa architecture. The model is trained to classify text as either human-written or AI-generated. You can download the model from [this link](https://staffsuniversity-my.sharepoint.com/:u:/g/personal/g029158m_student_staffs_ac_uk/EZppxmSEIu1DpeD8Thviu78B7uKtG60fQZP6q-LyQg-sSA?e=thugZh) and place it inside the folder 'roberta-base-openai-detector'.

## Installation
To run this application locally, make sure you have Python installed on your system. Then, follow these steps:
1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Install the required Python packages.
4. Run the application using the command `streamlit run app.py`.

## Usage
1. Upon launching the application, you will see a file uploader section where you can upload a PDF  file for analysis.
2. After uploading a file, the application will display the analysis results.
3. The Benford Law Analysis section shows whether the distribution of first letters in the text follows Benford's Law, indicating whether the text is likely human-written or AI-generated.
4. The Stylometric Analysis section provides insights into various linguistic features of the text, such as word length distribution, sentence length distribution, vocabulary richness, and punctuation usage.

## Dependencies
- Streamlit
- Numpy
- Matplotlib
- Pandas
- Scipy
- Transformers
- PyPDF
- Collections

## Contributors
- Janith Gomez

