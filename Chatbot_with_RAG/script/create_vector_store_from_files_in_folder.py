import boto3
import os
import uuid
from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# AWS Bedrock client setup
bedrock_client = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

def get_unique_id():
    return str(uuid.uuid4())

## Split the pages / text into chunks
def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

## Create a vector store
def create_vector_store(folder_name, documents):
    vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
    file_name = f"{folder_name}_vector_store"
    folder_path = os.getcwd()
    vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path+"/VectorIndexFiles")

    ## Optional: upload to S3 (commented out)
    # s3_client.upload_file(Filename=folder_path + "/" + file_name + ".faiss", Bucket=BUCKET_NAME, Key=f"{folder_name}_vector_store.faiss")
    # s3_client.upload_file(Filename=folder_path + "/" + file_name + ".pkl", Bucket=BUCKET_NAME, Key=f"{folder_name}_vector_store.pkl")

    return True

## Main method
def main():
    folder_path = "SampleFiles"
    combined_docs = []  # To store documents from all files

    # Process each PDF in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):  # Only process PDF files
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file: {file_path}")

            try:
                loader = PyPDFLoader(file_path)
                pages = loader.load_and_split()
                print(f"File '{file_name}' has {len(pages)} pages.")
                
                # Split Text
                splitted_docs = split_text(pages, chunk_size=1000, chunk_overlap=150)
                print(f"Splitted Doc length for '{file_name}': {len(splitted_docs)}")
                
                # Append to combined_docs
                combined_docs.extend(splitted_docs)
            except Exception as e:
                print(f"Error processing file '{file_name}': {e}")

    print("Creating the Vector Store for all documents in the folder...")
    result = create_vector_store("SampleFiles", combined_docs)

    if result:
        print("Vector Store created successfully for all PDFs in the folder.")
    else:
        print("Error creating the Vector Store.")

if __name__ == "__main__":
    main()
