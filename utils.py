# Importing necessary libraries
from sentence_transformers import SentenceTransformer  # For sentence embeddings
from openai import OpenAI
from dotenv import load_dotenv
import pinecone  # For vector database operations
import streamlit as st  # For creating web apps
import os

# Setting the API key for OpenAI
load_dotenv()
client = OpenAI(api_key = os.environ.get('OPENAI_API_KEY'))

# Initializing the sentence transformer model for generating embeddings
model = SentenceTransformer('all-mpnet-base-v2')

# Initializing Pinecone with the API key and setting the environment
pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment='gcp-starter')

# Connecting to an existing Pinecone index
index = pinecone.Index('tripadvisor-index')

# Function to find the best match for the input in the Pinecone index
def find_match(input):
    # Generating embedding for the input text
    input_em = model.encode(input).tolist()
    
    # Querying the Pinecone index with the generated embedding
    result = index.query(input_em, top_k=2, includeMetadata=True)

    # Add this line for debugging:
    print(f"Here is the result 1: {result['matches'][0]['metadata'].get('metadata')}")
    print(f"Here is the result 2: {result['matches'][1]['metadata'].get('metadata')}")
    
    # Returning the top 2 matched texts from the Pinecone index
    return result['matches'][0]['metadata'].get('metadata') + "\n" + result['matches'][1]['metadata'].get('metadata')

# Function to refine a user query based on the conversation context
def query_refiner(conversation, query):
    # Generating a refined query using OpenAI's GPT model
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
        temperature=0.1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # Accessing the text in the response
    refined_query = response.choices[0].text
    return refined_query

# Function to construct a string representation of the conversation history
def get_conversation_string():
    conversation_string = ""
    # Iterating through the conversation history stored in Streamlit's session state
    for i in range(len(st.session_state['responses'])-1):
        # Adding human and bot parts of the conversation to the string
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i+1] + "\n"
    return conversation_string
