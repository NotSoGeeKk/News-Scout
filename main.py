import os
import streamlit as st
import pickle
import time
import requests
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.schema import Document

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("Choose an Option")

# User selects the option
option = st.sidebar.radio(
    "How would you like to use this tool?",
    ("Process Specific URLs", "Search Current News")
)

file_path = "faiss_store.pkl"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad",temperature='0.3')
text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", ","], chunk_size=1000)


def process_urls(urls):
    """Process the URLs and return embeddings and FAISS index."""
    loader = UnstructuredURLLoader(urls=urls)
    st.info("Loading data from URLs...")
    data = loader.load()

    # Convert to LangChain Document objects
    docs = [Document(page_content=doc.page_content, metadata={}) for doc in data]
    split_docs = text_splitter.split_documents(docs)

    # Prepare embeddings
    embeddings = embedding_model.encode([doc.page_content for doc in split_docs])

    # Initialize FAISS index
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)

    documents = [
        {"content": doc.page_content, "metadata": doc.metadata} for doc in split_docs
    ]
    return faiss_index, documents


def fetch_news(query, api_key, max_results=5):
    """Fetch news articles using NewsAPI."""
    url = f"https://newsapi.org/v2/everything?q={query}&pageSize={max_results}&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json()["articles"]
        return [
            {
                "title": article["title"],
                "content": article["content"],
                "url": article["url"],
                "publishedAt": article["publishedAt"],
            }
            for article in articles
        ]
    else:
        st.error("Error fetching news articles.")
        return []


if option == "Process Specific URLs":
    urls = [st.sidebar.text_input(f"Enter URL {i+1}") for i in range(3)]
    process_clicked = st.sidebar.button("Process URLs")

    if process_clicked and any(urls):
        faiss_index, documents = process_urls(urls)
        with open(file_path, "wb") as f:
            pickle.dump({"index": faiss_index, "documents": documents}, f)
        st.success("URLs processed and indexed.")

elif option == "Search Current News":
    news_query = st.sidebar.text_input("Enter a topic to search for news articles:")
    api_key = st.sidebar.text_input("Enter your NewsAPI key:")

    if news_query and api_key:
        articles = fetch_news(news_query, api_key)
        if articles:
            st.subheader("Fetched Articles")
            for i, article in enumerate(articles):
                st.write(f"**{i+1}. {article['title']}**")
                st.write(f"[Read More]({article['url']})")
                st.write(f"Published At: {article['publishedAt']}")

            # Convert articles to LangChain Document objects
            texts = [article["content"] for article in articles if article["content"]]
            docs = [
                Document(page_content=text, metadata={"source": f"Article {i+1}"})
                for i, text in enumerate(texts)
            ]
            split_docs = text_splitter.split_documents(docs)

            # Prepare embeddings
            embeddings = embedding_model.encode([doc.page_content for doc in split_docs])

            # Initialize FAISS index
            dimension = embeddings.shape[1]
            faiss_index = faiss.IndexFlatL2(dimension)
            faiss_index.add(embeddings)

            documents = [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in split_docs
            ]
            with open(file_path, "wb") as f:
                pickle.dump({"index": faiss_index, "documents": documents}, f)
            st.success("Articles indexed.")

query = st.text_input("Ask a question:")
if query and os.path.exists(file_path):
    with open(file_path, "rb") as f:
        faiss_data = pickle.load(f)
        faiss_index = faiss_data["index"]
        documents = faiss_data["documents"]

        # Retrieve relevant documents
        embedding_query = embedding_model.encode([query])
        _, indices = faiss_index.search(embedding_query, 5)
        retrieved_docs = [documents[i] for i in indices[0]]

        # Prepare context for QA
        context = " ".join([doc["content"] for doc in retrieved_docs])
        result = qa_pipeline(question=query, context=context)

        # Display the result
        st.subheader("Answer")
        st.write(result["answer"])

        # Display sources
        st.subheader("Sources")
        for doc in retrieved_docs:
            st.write(doc["metadata"].get("source", "Unknown Source"))
