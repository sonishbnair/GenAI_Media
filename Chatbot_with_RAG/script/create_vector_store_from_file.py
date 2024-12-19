import boto3
import os
import uuid

## s3_client
#session = boto3.Session("viacbs-poc-dev")
#s3_client = session.client("s3")
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

## Bedrock
from langchain_community.embeddings import BedrockEmbeddings

## Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

## Pdf Loader
from langchain_community.document_loaders import PyPDFLoader

## import FAISS
from langchain_community.vectorstores import FAISS

bedrock_client = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

def get_unique_id():
    return str(uuid.uuid4())


## Split the pages / text into chunks
def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

## create vector store
def create_vector_store(file_name, documents):
    vectorstore_faiss=FAISS.from_documents(documents, bedrock_embeddings)
    file_name=f"{file_name}"
    folder_path=os.getcwd()
    vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)

    ## upload to S3
    # s3_client.upload_file(Filename=folder_path + "/" + file_name + ".faiss", Bucket=BUCKET_NAME, Key="my_faiss.faiss")
    # s3_client.upload_file(Filename=folder_path + "/" + file_name + ".pkl", Bucket=BUCKET_NAME, Key="my_faiss.pkl")

    return True

## main method
def main():
    file_name = "SampleFiles/WhatsNew_MediaComposer_v22.12.pdf"
    file_path = file_name
    #file_path = os.getcwd()+"/SampleFiles/"+file_name
    print("File path -- "+file_path)
    # with open(file_path, mode="wb") as w:
    #     w.write(uploaded_file.getvalue())

    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()

    print("Total Pages: {len(pages)}")

    ## Split Text
    splitted_docs = split_text(pages, 1000, 150)
    print("*******************")
    print("Splitted Doc length: "+str(len(splitted_docs)))
    print("*******************")
    #print(splitted_docs[0])

    print("Creating the Vector Store")
    result = create_vector_store(file_name, splitted_docs)

    if result:
        print("Vector Embbeded created for the PDF successfully")
    else:
        print("Error!! Craeting the Vector")

if __name__ == "__main__":
    main()