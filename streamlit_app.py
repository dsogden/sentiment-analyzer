import streamlit as st
import transformers


st.title("Sentiment Analysis Application")
# st.write(
#     "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
# )

tokenizer = transformers.DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

model = transformers.DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

st.write("Enter a text below to analyze its sentiment:")
user_input = st.text_area("Text to analyze", placeholder="Type something...")
if st.button('Analyze Sentiment'):
    encoded = tokenizer(user_input, return_tensors="pt")
    output = model(**encoded)
    predicted_class_id = output['logits'].argmax().item()
    prediction = model.config.id2label[predicted_class_id]
    st.write(prediction)