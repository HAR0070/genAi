import streamlit as st 
from qdrant_client import QdrantClient, models

import io
import uuid
import os

from collections import namedtuple
import numpy as np
import json
from urllib.parse import urljoin

import requests
import lxml.html.clean
from trafilatura import fetch_url, extract 
from trafilatura.settings import use_config

from bs4 import BeautifulSoup
from docx import Document
from PIL import Image
from pydantic import ValidationError

from langchain_community.document_loaders import PyMuPDFLoader
import tempfile
from sentence_transformers import SentenceTransformer
from google import genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as document_format



# config parameter for fetch_url
new_config = use_config()
new_config.set("DEFAULT" , "EXTRACTION_TIMEOUT", "10")

# to clean the error mesasges after each iter
status_placeholder = st.empty()

st.title("Your Personal Gemini RAG Chatbot 💬")

# GEMINI_API_KEY="AIzaSyCd_S7ShXDMg0KUYpOxcmBxWBGcbAnJtQo"
# Qdrant_API_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.kxkquuJDRrStU1OIKKbNlOCCneBpXyS0TPiosiC4ue4"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
Qdrant_API_KEY = os.getenv("Qdrant_API_KEY")

QDRANT_URL = "https://6e71e5b7-240f-48b5-aff6-7cf6ecf8009d.us-east-1-1.aws.cloud.qdrant.io:6333"


@st.cache_resource
def get_qdrant_client(url):
    return QdrantClient( url, api_key=Qdrant_API_KEY)

q_drant_client = get_qdrant_client(QDRANT_URL)

@st.cache_resource
def get_gemini_client(api_key=GEMINI_API_KEY, model="gemini-2.5-flash" ):
    genai_client = genai.Client(api_key = api_key )
    return genai_client.chats.create(model=model)

@st.cache_resource
def get_text_model(name="BAAI/bge-base-en-v1.5"):
    return SentenceTransformer(name)

@st.cache_resource
def get_image_model(name="clip-ViT-L-14"):
    return SentenceTransformer(name)

class Chatbot:
    def __init__(self, GEMINI_API_KEY,qdrant_client , base_name):
        self.qdrant_client = qdrant_client
        self.hits = []
        self.context = []
        self.collection_name = base_name
        self.model_name = get_text_model()
        #initialize gemini client
        self.genai_client = genai.Client(api_key = GEMINI_API_KEY )
        self.chat = self.genai_client.chats.create(model= "gemini-2.5-flash")
        
        #get_gemini_client()

        # Initialize Qdrant collection - self.collection is a boolean
        self.collection = qdrant_client.create_collection(
            collection_name= self.collection_name,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
        )

        #initialize context
        self.system_prompt = "You are a helpful, humble, genius, careful, thoughtFull and cautious, assistant. you look deeply into context, If the answer isn't in the context, you specify whats not there, and try to search your own knowledge base to find most appropriate answer according to the context."

        # Initialize chat history  if required
        # self.messages = [{"role": "system", "content": self.system_prompt}]
    
    def query(self , prompt , n_points) -> str :

        try:
            results = self.qdrant_client.query_points( collection_name= self.collection_name,
                    query= self.model_name.encode(prompt, convert_to_numpy=True).tolist(),
                    limit=n_points)
        except ValidationError as exc:
            status_placeholder.warning(f"this is promt type {repr(exc.errors()[0]['type'])} ")
            return " "  # return empty context on error
    
        context = "\n".join(r.payload['document'] for r in results.points)

        return context
        
    def query_genai(self , prompt):
        
        context = self.query(prompt , n_points=25)
        
        # single input that contains system + context + user question
        input_message = f"{self.system_prompt}\n\nRAG CONTEXT:\n{context}\n\nQuestion: {prompt}"

        response = self.chat.send_message_stream(input_message)

        if response:
            for chunk in response:
                st.write(chunk.text)
        else:
            status_placeholder.info("Chatbot didnt reply")
            fault_msg = "I am kinda busy - sorry cant process now"
            response = st.write(fault_msg)

class Database():
    
    def __init__(self, qdrant_client , base_name):
        self.qdrant = qdrant_client
        self.collection_name = base_name
        self.model_name = get_text_model()
        
        self.image_model = get_image_model() 
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=768, chunk_overlap=30)
                
    def extract_url(self , url):
        text_img= namedtuple('text_img', ['text_json', 'image_links'])
        image_url = []
        try:
            downloaded = fetch_url(url, config=new_config)
            if not downloaded:
                status_placeholder.error("Hey can you check the URL please, I am having some trouble with it.")
                return None
            else:
                kwargs = { "output_format": "json",
                        "with_metadata": True,  # If meta data is req - then use json type
                        "include_images": True}

                raw = extract(downloaded, **kwargs) # this is the text content
                soup = BeautifulSoup(downloaded, "html.parser")  # this is to parse HTML images 
                
                if not raw:
                    status_placeholder.error("The webpage is empty or could not be processed.")
                    return None
                else:
                    for img in soup.find_all('img'):
                        src = img.get("src") or img.get("data-src") or img.get("data-original") or img.get("data-lazy")
                        if not src:
                            continue
                        full = urljoin(url, src)                 # make absolute
                        image_url.append(full)
        
            return text_img(text_json=raw, image_links=image_url)
        
        except Exception as e:
            status_placeholder.error(f"An error occurred while fetching the URL: {e}")
            return None

    def fetch_image(self, url):
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()  # Raise an error for bad responses

            img = Image.open(io.BytesIO(response.content)).convert("RGB")
            return self.image_model.encode(img, convert_to_numpy=True)
        
        except Exception as e:
            status_placeholder.error(f"Error fetching image from {url}: {e}")
            return None
        
    def add_text(self, data, source):
        chunks = self.splitter.split_text(data)
                
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector= self.model_name.encode(chunk, convert_to_numpy=True).tolist(),
                payload={"document" : chunk, "source": source}
            )
            for chunk in chunks
        ]
        
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def add_image_from_web(self, url_stack, source):
        
        for url in url_stack:
            image_embeding = self.fetch_image(url)
            if image_embeding is not None:
                points = []
                points.append( models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=image_embeding.tolist(),
                    payload={"document" : self.fetch_image(url) ,"type": "image", "image_url": url, "source": source}
            ))
                self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points)
            else:
                # status_placeholder.warning(f"Skipping image from {url} due to fetch error.")
                continue
        
    def add_image_input(self, img_file , source):
        img_raw = img_file.read()
        image_embeding = None
        try:
            img = Image.open(io.BytesIO(img_raw)).convert("RGB")
            image_embeding = self.image_model.encode(img, convert_to_numpy=True)
            points = []
            points.append( models.PointStruct(
                id=str(uuid.uuid4()),
                vector=image_embeding.tolist(),
                payload={"document" : img_raw , "type": "image", "source": source}
                ))
            self.qdrant.upsert(
            collection_name=self.collection_name,
            points=points)
        except Exception as e:
            status_placeholder.error(f"Error using this image {source}: {e}")
            return None
        
    def Push_website_to_qdrant(self, url_data):
        data_tupple = self.extract_url(url_data)
        if data_tupple:
            data = json.loads(data_tupple.text_json)
            self.add_text(data["text"] , source=data["title"])

            if data_tupple.image_links:
                self.add_image_from_web(data_tupple.image_links , source=data["title"])
                status_placeholder.success("URL content indexed successfully.")

    def Push_pdf_to_qdrant(self, doc_file):
        
        if doc_file.type == "application/pdf" :
            suffix = os.path.splitext(doc_file.name)[1] or ".pdf"
        else:
            suffix = os.path.splitext(doc_file.name)[1] or ".docx"
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(doc_file.read())
            tmp_path = tmp.name

        try:
            # 1. Use the dedicated PyMuPDFLoader
            langchain_docs = []
            if doc_file.type != "application/pdf":
                loader = Document(tmp_path)
                for i, para in enumerate(loader.paragraphs):
                    # Create the Document object
                    doc = document_format(
                        page_content= para.text,
                        metadata={
                            "source": doc_file.name,
                            "para": i + 1, # para numbers are typically 1-indexed
                        }
                    )
                    langchain_docs.append(doc)
                    
            else:
                loader = PyMuPDFLoader(tmp_path)  # confirm loader accepts file-like
                langchain_docs = loader.load()

            # 2. Load the documents - this returns a list of LangChain Document objects
            chunked_docs = self.splitter.split_documents(langchain_docs)
            for chunks in chunked_docs:
                self.add_text(chunks.page_content , source= tmp_path)
        
        finally:
            # cleanup
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    def service_all_docs(self, uploaded_files):
        if uploaded_files:
            for files in uploaded_files:
                if files.type.startswith("image/"):
                    self.add_image_input(files, source=files.name)
                    status_placeholder.success(f"Image {files.name} indexed successfully.")
                    
                elif files.type in [ "application/pdf" , "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                    
                    self.Push_pdf_to_qdrant(files)
                    status_placeholder.success(f"PDF {files.name} indexed successfully.")
                    
                elif files.type == "text/csv":
                    import pandas as pd
                    df = pd.read_csv(files)
                    text = ""
                    if 'text' in df.columns:
                        for index, row in df.iterrows():
                            text = text + str(row['text'])
                            if len(text) >= 768:  # chunk size limit
                                self.add_text(text[:768] , source=files.name)
                                text = text[768:].strip()  # retain leftover text
                        status_placeholder.success(f"CSV {files.name} indexed successfully.")
                    else:
                        status_placeholder.info(f"CSV {files.name} does not contain a 'text' not good idea to infer from numbers alone.")
                else:
                    status_placeholder.error(f"Unsupported file type: {files.type}")

# The `chatbot_` class is responsible for creating a chatbot that integrates both the Gemini API and
# the Qdrant client. It initializes the GPT model for chat responses, sets up the Qdrant collection
# for storing and querying data, and handles the interaction between the user input, context retrieval
# from Qdrant, and generating responses using the Gemini model. The `making_context` method retrieves
# and formats the context from Qdrant search results, the `query` method queries the Qdrant collection
# for relevant context based on the user prompt, and the `query_genai` method combines the Qdrant
# context with the user prompt to generate a response using the Gemini model.


if "name" not in st.session_state or not st.session_state.name:
    name = st.chat_input("How should i Address you?")
    if name:
        st.session_state.name = name
        st.rerun()
    else:
        status_placeholder.warning("I am waiting ......")
        st.stop()
        

if "base" not in st.session_state:
        unique_id = uuid.uuid4().hex[:8]   # short unique token
        st.session_state.base = f"{st.session_state.name}_rag_{unique_id}"

if "chatbot_" not in st.session_state:
    st.session_state.chatbot_ = Chatbot(GEMINI_API_KEY , q_drant_client, st.session_state.base )

if "database" not in st.session_state:
    st.session_state.database = Database(q_drant_client , st.session_state.base)

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.session_state.uploaded_files = st.file_uploader("Upload data", accept_multiple_files=True,
                type=["jpg", "csv", "pdf", "docx",  "png"])

if st.session_state.uploaded_files:
    st.session_state.database.service_all_docs(st.session_state.uploaded_files)
    del st.session_state.uploaded_files 
    status_placeholder.empty()

web_page = st.text_input("Enter article URL", "")

if st.button("Fetch and Process URL") and web_page.strip():
    status_placeholder.info("Fetching and processing the URL...")
    st.session_state.database.Push_website_to_qdrant(web_page)
    status_placeholder.empty()

# Accept user input
if prompt := st.chat_input("What is up?"):

    # Display user message in chat message container
    with st.chat_message(st.session_state.name):
        st.markdown(prompt)
        

# Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.session_state.chatbot_ .query_genai(prompt)



