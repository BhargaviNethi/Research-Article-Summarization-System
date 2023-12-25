import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64
from gtts import gTTS  # Import the gTTS library for text-to-speech

#MODEL and TOKENIZER
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
based_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map = 'auto', torch_dtype = torch.float32)

#file loader and preprocessing
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts= ""
    for text in texts:
        print(text)
        final_texts = final_texts + text.page_content
    return final_texts

#LM pipeline
def llm_pipeline(filepath, max_length=1000, min_length=50):
    pipe_sum = pipeline(
        'summarization',
        model = based_model,
        tokenizer = tokenizer,
        max_length = max_length,
        min_length = min_length
    )
    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text)
    result = result[0]['summary_text']
    audio_summary = gTTS(result)

    # Save the audio summary as an MP3 file
    audio_summary.save("summary.mp3")
    return result

@st.cache_data
#function to display the pdf of the given file 
def displayPDF(file):
    #Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    
    #embedding pdfin html
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    #Displaying file
    st.markdown(pdf_display, unsafe_allow_html=True)

#streamlit code
st.set_page_config(layout='wide')

def main():

    st.title('RESEARCH ARTICLE SUMMARIZER')

    uploaded_file = st.file_uploader("Upload your PDF File", type=['pdf'])

    if uploaded_file is not None:
        summary_length = st.selectbox("Select Summary Length", ["Small", "Medium", "Large"])
        if st.button("Summarizer"):
            col1,col2,col3 = st.columns(3)
            filepath = "data/"+uploaded_file.name
            with open(filepath, 'wb') as temp_file:
                temp_file.write(uploaded_file.read())
            with col1:
                st.info("Uploaded PDF File")
                pdf_viewer = displayPDF(filepath)


            with col2:
                st.info("Summarization is below")

                if summary_length == "Small":
                    max_length = 200
                    min_length = 20
                elif summary_length == "Medium":
                    max_length = 400
                    min_length = 40
                else:
                    max_length = 600
                    min_length = 60

                summary = llm_pipeline(filepath, max_length=max_length, min_length=min_length)
                st.success(summary)
            
            with col3:
                st.info("Read Aloud")
                st.audio("summary.mp3", format="audio/mp3")

if __name__ == '__main__':
    main()