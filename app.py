import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from dotenv import load_dotenv
from langchain.docstore.document import Document
import textwrap

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_question_generation_chain():
    prompt_template = """
    Generate a list of questions from the provided context. Make sure the questions are specific and answerable based on the context.
    Context:\n {context}\n
    Questions:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, convert_system_message_to_human=True)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def generate_questions(text_chunks):
    chain = get_question_generation_chain()
    questions = []
    for chunk in text_chunks:
        doc = Document(page_content=chunk)
        response = chain({"input_documents": [doc]}, return_only_outputs=True)
        questions.extend(response["output_text"].split("\n"))
    return questions

def filter_questions_with_answers(questions, text_chunks):
    qa_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, convert_system_message_to_human=True)
    qa_chain = load_qa_chain(qa_model, chain_type="map_reduce")
    filtered_questions = []

    for question in questions:
        docs = [Document(page_content=chunk) for chunk in text_chunks]
        response = qa_chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        if response["output_text"].strip():  # Check if the answer is not empty
            filtered_questions.append(question)

    return filtered_questions

def create_pdf(questions, file_path):
    c = canvas.Canvas(file_path, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica", 12)
    margin = 40
    max_width = width - 2 * margin

    y = height - margin
    for question in questions:
        if y < margin:
            c.showPage()
            c.setFont("Helvetica", 12)
            y = height - margin
        wrapped_text = textwrap.fill(question, width=int(max_width / 7))  # Adjust the width as needed
        lines = wrapped_text.split("\n")
        for line in lines:
            if y < margin:
                c.showPage()
                c.setFont("Helvetica", 12)
                y = height - margin
            text = c.beginText(margin, y)
            text.textLine(line)
            c.drawText(text)
            y -= 14  # Adjust line spacing as needed

    c.save()

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    qa_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, convert_system_message_to_human=True)
    qa_chain = load_qa_chain(qa_model, chain_type="map_reduce")

    response = qa_chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(
        page_title="PDF QA with Gemini",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.header("Chat with PDF Using the Power of Gemini🤖🚀")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.sidebar.title("Upload PDF File(s)")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            try:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success('File Submitted and Processed Successfully')
            except Exception as e:
                st.error(f"Error during processing: {e}")
        
        if st.button("Generate Question Paper"):
            try:
                with st.spinner("Generating Question Paper..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    questions = generate_questions(text_chunks)
                    questions_with_answers = filter_questions_with_answers(questions, text_chunks)
                    question_paper_path = "question_paper.pdf"
                    create_pdf(questions_with_answers, question_paper_path)
                    st.success('Question Paper Generated Successfully')
                    st.download_button("Download Question Paper", data=open(question_paper_path, "rb"), file_name="question_paper.pdf")
            except Exception as e:
                st.error(f"Error generating question paper: {e}")

if __name__ == "__main__":
    main()