import logging
import streamlit as st
from datasets import load_dataset
from transformers import pipeline

MAX_LENGTH = 25
ENGLISH_VARIETIES = ["UK/British English", "Indian English", "Australian English", "All Varieties"]

# Logger
log_level = "INFO"
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)
logger.setLevel(log_level)

# Fetch some examples
@st.cache_data
def get_example_data():
    ds = load_dataset("surrey-nlp/BESSTIE-CW-26")
    return ds

# Load Models
@st.cache_resource
def load_sarcasm_au_model():
    sarc_au_pipeline = pipeline("text-classification", model="boxallcharlie/Sarcasm-Best-Test-Variety-AU")
    return sarc_au_pipeline

@st.cache_resource
def load_sarcasm_uk_model():
    sarc_uk_pipeline = pipeline("text-classification", model="boxallcharlie/Sarcasm-Best-Test-Variety-UK")
    return sarc_uk_pipeline

@st.cache_resource
def load_sarcasm_in_model():
    sarc_in_pipeline = pipeline("text-classification", model="boxallcharlie/Sarcasm-Best-Test-Variety-IN")
    return sarc_in_pipeline

@st.cache_resource
def load_sarcasm_all_model():
    sarc_all_pipeline = pipeline("text-classification", model="boxallcharlie/Sarcasm-Best-Test-Variety-ALL")
    return sarc_all_pipeline

@st.cache_resource
def load_sentiment_au_model():
    sent_au_pipeline = pipeline("text-classification", model="boxallcharlie/Sentiment-Best-Test-Variety-AU")
    return sent_au_pipeline

@st.cache_resource
def load_sentiment_uk_in_model():
    # UK and IN share the same best model
    sent_uk_in_pipeline = pipeline("text-classification", model="boxallcharlie/Sentiment-Best-Test-Variety-UK-and-IN")
    return sent_uk_in_pipeline

@st.cache_resource
def load_sentiment_all_model():
    sent_all_pipeline = pipeline("text-classification", model="boxallcharlie/Sentiment-Best-Test-Variety-ALL")
    return sent_all_pipeline
   
# Define helper functions
def predict_sarcasm(text, english_variety):
    """
    Selects the optimal model based on english_variety input and makes a prediction.

    Parameters:
    text: Input by user.
    english_variety: String input that must be either "UK/British English", "Indian English", "Australian English"

    Returns: model inference result
    """
    if english_variety == "UK/British English":
        model = load_sarcasm_uk_model()
    elif english_variety == "Indian English":
        model = load_sarcasm_in_model()
    elif english_variety == "Australian English":
        model = load_sarcasm_au_model()
    elif english_variety == "All Varieties":
        model = load_sarcasm_all_model()
    else:
        raise ValueError(f"Unknown English variety: {english_variety}")

    result = model(text)
    logger.info(f"Prediction for '{text[:50]}...' using {english_variety}: {result}")
    raw_label = result[0]["label"].lower()
    if raw_label in ("label_0", "not-sarcastic", "not sarcastic"):
        label = "not sarcastic"
    elif raw_label in ("label_1", "sarcastic"):
        label = "sarcastic"
    else:
        raise ValueError(f"Unknown label from result: {result[0]['label']}")

    return result, label

def predict_sentiment(text, english_variety):
    """
    Selects the optimal model based on english_variety input and makes a prediction.

    Parameters:
    text: Input by user.
    english_variety: String input that must be either "UK/British English", "Indian English", "Australian English"

    Returns: model inference result
    """
    if english_variety == "UK/British English":
        model = load_sentiment_uk_in_model()
    elif english_variety == "Indian English":
        model = load_sentiment_uk_in_model()
    elif english_variety == "Australian English":
        model = load_sentiment_au_model()
    elif english_variety == "All Varieties":
        model = load_sentiment_all_model()
    else:
        raise ValueError(f"Unknown English variety: {english_variety}")

    result = model(text)
    logger.info(f"Prediction for '{text[:50]}...' using {english_variety}: {result}")
    raw_label = result[0]["label"].lower()
    if raw_label in ("label_0", "negative"):
        label = "negative"
    elif raw_label in ("label_1", "positive"):
        label = "positive"
    else:
        raise ValueError(f"Unknown label from result: {result[0]['label']}")

    return result, label


def main():

    st.sidebar.title("Created By:")
    st.sidebar.subheader("Charlie Boxall, Divya Bisht, Zhong Han Loo, Jackie Malooly, Rishiraj Tripathi")
    st.sidebar.subheader("University of Surrey")
    st.sidebar.subheader("COMM061 Natural Language Processing Module")

    st.title("Sarcasm & Sentiment Analyzer")

    # Create tabs
    tab1, tab2 = st.tabs(["Sarcasm Detection", "Sentiment Analysis"])

    # Tab 1: Sarcasm Detection
    with tab1:
        st.header("Sarcasm Detector")

        with st.form(key='sarcasm_form'):
            english_variety = st.selectbox("Select the variety of English:", ENGLISH_VARIETIES, help="Select a variety of English for detecting Sarcasm. At this time our models are only optimised for these three varieties.")
            raw_text = st.text_area("Enter your text below:")

            submit_text = st.form_submit_button(label='Submit')

        if submit_text:

            if raw_text == "":
                st.exception(NameError("Please enter some text to test."))

            else:
                col1, col2 = st.columns(2)

                prediction, label = predict_sarcasm(raw_text, english_variety)
                with col1:
                    st.success("Your input")
                    st.write(raw_text)
                    st.success("Prediction")
                    st.write(label)
                with col2:
                    st.success("Probability")
                    st.write(float(prediction[0]["score"]))

        ds = get_example_data()
        sarcasm_examples = [ex for ex in ds['validation'] if ex['Sarcasm'] == 1.0][:3]
        neg_sarcasm_examples = [ex for ex in ds['validation'] if ex['Sarcasm'] == 0.0][:3]

        st.subheader("Sample predictions:")
        st.success(f"{sarcasm_examples[0]["text"]}\n-sarcastic\n\n{sarcasm_examples[1]["text"]}\n-sarcastic")
        st.success(f"{neg_sarcasm_examples[0]["text"]}\n-not sarcastic\n\n{neg_sarcasm_examples[1]["text"]}\n-not sarcastic")

    # Tab 2: Sentiment Analysis
    with tab2:
        st.header("Sentiment Analyzer")

        with st.form(key='sentiment_form'):
            sent_variety = st.selectbox("Select the variety of English:", ENGLISH_VARIETIES, help="Select a variety of English for sentiment analysis.")
            sent_text = st.text_area("Enter your text below:")

            submit_sentiment = st.form_submit_button(label='Submit')

        if submit_sentiment:
            if sent_text == "":
                st.exception(NameError("Please enter some text to test."))
            else:
                col1, col2 = st.columns(2)

                sent_prediction, sent_label = predict_sentiment(sent_text, sent_variety)
                with col1:
                    st.success("Your input")
                    st.write(sent_text)
                    st.success("Prediction")
                    st.write(sent_label)
                    st.write(sent_prediction)
                with col2:
                    st.success("Probability")
                    st.write(float(sent_prediction[0]["score"]))

        ds = get_example_data()
        sarcasm_examples = [ex for ex in ds['validation'] if ex['Sentiment'] == 1.0][:3]
        neg_sarcasm_examples = [ex for ex in ds['validation'] if ex['Sentiment'] == 0.0][:3]

        st.subheader("Sample predictions:")
        st.success(f"{sarcasm_examples[0]["text"]}  \n-positive    \n  \n{sarcasm_examples[1]["text"]}  \n-positive")
        st.success(f"{neg_sarcasm_examples[0]["text"]}    \n-negative  \n  \n{neg_sarcasm_examples[1]["text"]}  \n-negative")

if __name__ == "__main__":
    main()