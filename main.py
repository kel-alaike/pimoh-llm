# Import necessary libraries and modules for the chatbot
import streamlit as st
import sys
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from streamlit_chat import message
from utils import *

sys.path.append('.')

# Define API keys for OpenAI and Pinecone
OPENAI_API_KEY = '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'
PINECONE_API_KEY = 'e20aa772-b33e-478e-a86d-31e5eb80d9dc'

# Set up a Streamlit header for the application
st.subheader("Pimoh EatBot 9100 🦩")

# Initialize session states for storing responses and requests
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Hey, where are you thinking of eating?"]
if 'requests' not in st.session_state:
    st.session_state['requests'] = []

# Initialize the ChatOpenAI model with GPT-4
llm = ChatOpenAI(model_name="gpt-4-1106-preview", openai_api_key=OPENAI_API_KEY)

# Set up a conversation buffer memory for storing recent messages
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory=ConversationBufferWindowMemory(k=10, return_messages=True)

# Define templates for system and human messages in the conversation
system_msg_template = SystemMessagePromptTemplate.from_template(template="""
    As the user inquires about restaurant recommendations, your task is to guide them using the 7 Ws approach:
                                                                
    You will only have 3 chances to give the user their 3 recommendations so make your responses count

    1. **Who**: "Will you be dining alone, or is this for a group? Are there any children?"
    2. **What**: "Are you looking for a specific type of cuisine or dish?"
    3. **When**: "Do you have a particular time or day in mind for your visit?"
    4. **Where**: "Any specific area or location you prefer?"
    5. **Why**: "Is this for a special occasion or just a casual outing?"
    6. **Which**: "Do you have any specific requirements like vegan options or a kid-friendly environment?"
    7. **How**: "What's your preferred price range or ambiance - casual, upscale, etc.?"

    Respond conversationally as if speaking to a friend, keeping answers to about 60 words or less, with a new paragraph every two sentences for readability. 
    If unsure, ask clarifying questions to get the user back on track.
""")

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

# Create a chat prompt template combining system and human message templates
prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

# Set up the conversation chain with memory, prompt template, and the language model
conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

# Define Streamlit containers for chat history and text input
response_container = st.container()
textcontainer = st.container()

# Handle user input and generate responses
with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("thinking..."):
            conversation_string = get_conversation_string()
            refined_query = query_refiner(conversation_string, query)
            st.subheader("Refined Query:")
            st.write(refined_query)
            context = find_match(refined_query)
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response) 

# Display chat history in the Streamlit app
with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')
