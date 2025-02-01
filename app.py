import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("model\checkpoint-372")
tokenizer = AutoTokenizer.from_pretrained("model\checkpoint-372")

def summarize_text(inputs):
    # Summarize the provided text
    
    summary_ids = model.generate(
    inputs['input_ids'],  # Input tokens
    max_length=150,  # Maximum length of the output sequence
    num_beams=4,  # Beam search to improve output quality
    no_repeat_ngram_size=2,  # Avoid repeating n-grams
    length_penalty=2.0,  # Control length of output
    early_stopping=True  # Stop as soon as we reach the maximum length
    )

    # Decode and print the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit UI setup
st.title("Text Summarizer")
st.write("Enter the text you want to summarize:")

# Text input area
input_text = st.text_area("Enter your text here", height=300)
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
if st.button("Generate Summary"):
    if inputs:
        with st.spinner("Summarizing..."):
            summary = summarize_text(inputs)
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.warning("Please enter some text to summarize.")

