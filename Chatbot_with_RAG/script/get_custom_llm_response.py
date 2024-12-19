import boto3
import os
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Initialize Bedrock Client and Embeddings
bedrock_client = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client) 
# To-do: Also try the model - amazon.titan-text-express-v1

# Vector path
folder_path = "./VectorIndexFiles"
print("folder_path --- " + folder_path)

def connect_llm():
    return Bedrock(
        model_id="amazon.titan-text-lite-v1",
        client=bedrock_client,
        model_kwargs={'maxTokenCount': 500}  # Adjusted token count
    )
    # return Bedrock(
    #     model_id="amazon.nova-micro-v1:0",
    #     client=bedrock_client,
    #     model_kwargs={'maxTokenCount': 1000}  # Adjusted token count
    # )

def get_response_from_llm(llm, vectorstore, question):
    # Create prompt/template
    context = """You are an expert assistant for media professionals such as video/media engineers, editors, archivists, and media managers.
    When you respond, do the following:
    1. Summarize and explain the context in simple terms.
    2. Add relevant details or related insights.
    3. Avoid repeating text verbatim unless necessary for clarity.
    4. If the context includes an IP address, explain its role or purpose in the system.
    5. Do not ask follow-up question in the response, respond with available information.
    6. Give a response minimum of 200 chatacters.
    If you don't know the answer, suggest asking related questions in the domain of CBS Production Services, PAM, MAM, or Avid systems."""

    prompt_template = """
    Human: Please use the given context to provide answer to the question. Do not ask follow-up questions.
    Create a response from what you understood from the question.
    
    <context>
    {context}
    </context>
    
    Question: {question}
    Assistant:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        #To-do: Try chain_type="map_reduce" option as well.
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    answer = qa({"query": question})
    # print("**********")
    # print(answer)
    # print("**********")
    return answer['result']

def main():
    try:
        dir_list = os.listdir(folder_path)
        print("Vector files found:", dir_list)
        
        # Load vector store
        vector_store_index = FAISS.load_local(
            index_name="CBS_Custom_Vector_Store",
            folder_path=folder_path,
            embeddings=bedrock_embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Get question input from terminal
        print("====================")
        print("")
        question = input("Enter your question: ")
        
        llm = connect_llm()
        llm_response = get_response_from_llm(llm, vector_store_index, question)

        print("====================")
        print("Question:", question)
        print("====================")
        print("")
        print("LLM Response:")
        print(llm_response)
        print("\n")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
