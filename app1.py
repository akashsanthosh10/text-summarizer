import streamlit as st
from transformers import pipeline

# Load and cache the summarizer model
@st.cache_resource
def load_model():
    # Initialize the summarizer pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer

# Streamlit App UI
def main():
    # Load the model
    summarizer = load_model()

    # App title and description
    st.title("AI Text Summarizer")
    st.subheader("Effortlessly summarize long articles, documents, or any text into concise summaries.")
    st.write("Simply input a lengthy text below, and our AI will condense it into a concise and meaningful summary.")

    # Input text box for the user
    input_text = st.text_area("Enter text to summarize:", height=250)

    # Check if input text is provided
    if input_text.strip() == "":
        st.warning("Please enter some text to summarize.")
        return

    # Sliders for summary length
    min_length = st.slider("Minimum summary length (in words):", 10, 200, 50)
    max_length = st.slider("Maximum summary length (in words):", 50, 500, 150)

    # Button to trigger the summarization
    if st.button("Generate Summary"):
        with st.spinner("Summarizing... Please wait."):
            summary = summarizer(input_text, max_length=max_length, min_length=min_length, do_sample=False)
        
        # Display the summary
        st.subheader("Generated Summary:")
        st.write(summary[0]['summary_text'])

    # Footer with a simple note
    st.markdown("---")

if __name__ == "__main__":
    main()
