# Description: This file contains the main code for the Agentic OpenAI Chatbot. 
# The chatbot is powered by OpenAI's engine and uses Azure Cognitive Search to 
# search for documents and Bing Web Search to search the web for information.
# The chatbot can also provide the current temperature for a given location.
# The code defines functions for document search, web search, and getting weather information.
# The response function processes the user input and calls the appropriate function based on the input.
# The Streamlit code handles the user interface and interaction with the chatbot.

from openai import OpenAI
import os
import json
import streamlit as st
import requests
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from openai import AzureOpenAI
from datetime import datetime
import time

# Get the current date and time
current_datetime = datetime.now()

# Extract the current year
current_year = current_datetime.year

# Format the current date (optional)
current_date = current_datetime.strftime("%Y-%m-%d")  # Format as YYYY-MM-DD

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI()

llm_client = AzureOpenAI(
  api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version = "2024-06-01",
  azure_endpoint =os.getenv("AZURE_OPENAI_ENDPOINT") 
)

search_client = SearchClient(
    endpoint= os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT"),
    index_name= os.getenv("AZURE_SEARCH_INDEX"),
    key= os.getenv("AZURE_SEARCH_API_KEY"),
    credential=AzureKeyCredential(str(os.getenv("AZURE_SEARCH_API_KEY")))
 )

def document_search(query=None):
        # Step 1: Generate Embeddings
        embeddings = llm_client.embeddings.create(
            input=query,
            model="text-embedding-3-large"
        )
        embedding_vector = embeddings.data[0].embedding  # Extract the embedding vector

        # Step 2: Perform Hybrid Search
        search_results = search_client.search(
            search_text=query,  # Full-text search query
            vector_queries=[  # Vector query
                {
                    "kind": "vector",  # Specify the kind of query
                    "vector": embedding_vector,  # Embedding vector
                    "fields": "text_vector",  # Field in the index containing embeddings
                    "k": 10  # Number of nearest neighbors to retrieve
                }
            ],
            select=["title", "chunk"],  # Fields to include in the results
            top=10,  # Maximum number of results to return
            query_type="semantic",  # Enable semantic search
            semantic_configuration_name=os.getenv("AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG")  # Semantic config name
        )

        # Step 3: Process Results
        results = []
        for result in search_results:
            results.append(result)

        return json.dumps(results)

def web_search(query=None):
        endpoint = os.getenv('BING_ENDPOINT')
        subscription_key = os.getenv('BING_KEY')
        headers = {'Ocp-Apim-Subscription-Key': subscription_key}
        mkt = 'en-US'
        count = '10'
        freshness = "Week"
        params = { 'q': str(query), 'mkt': mkt , 'count': count, 'freshness': freshness}
        web_search_result = requests.get(endpoint, headers=headers, params=params)
    
        # Check if the request was successful
        if web_search_result.status_code == 200:
            # Convert the JSON response to a string and return it
            return json.dumps(web_search_result.json())
        else:
            # If the request failed, return an error message as a string
            return f"Error: Unable to fetch data. Status code: {web_search_result.status_code}"

def get_weather(latitude, longitude):
    weather = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
    data = weather.json()
    return data['current']['temperature_2m']

tools = [
    {"type": "function", 
     "function": {
        "name": "document_search",
        "description": "Search for documents using Azure Cognitive Search.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query."
                }
            },
            "required": ["query"],
            "additionalProperties": False
        },
        "strict": False
    }
    },
    {"type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web using Bing Web Search.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query."
                }
            },
            "required": ["query"],
            "additionalProperties": False
        },
        "strict": False
    }
    },
    {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number"},
                "longitude": {"type": "number"}
            },
            "required": ["latitude", "longitude"],
            "additionalProperties": False
        },
        "strict": False
    }
    }
]

def response(input):
     messages=[{
        "role": "developer", 
        "content": "You are a helpful assistant. The current date is " + current_date
        },
        {
        "role": "user", 
        "content": input
        }]
     completion = client.chat.completions.create(
          model="gpt-4o-mini",
          messages=messages,
          tools=tools,
          tool_choice="required"
          )
     tool_call = completion.choices[0].message.tool_calls[0]
     tool_call_name = tool_call.function.name
     args = json.loads(tool_call.function.arguments)
     if tool_call_name == "get_weather":
         results = get_weather(args["latitude"], args["longitude"])
     elif tool_call_name == "web_search":
         results = web_search(args["query"])
     elif tool_call_name == "document_search":
         results = document_search(args["query"])  # Call the function with the query
     
     # Display the function being executed in the sidebar
     st.sidebar.write("Last executed function: " + tool_call_name, args)

    # Create a new message with the tool call
     messages.append({
            "role": "assistant",
            "tool_calls": [tool_call.model_dump()]
     })
     messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(results)
     })
     final_response = client.chat.completions.create(
           model="gpt-4o-mini",
           messages=messages,
           tools=tools,
     )
     return final_response.choices[0].message.content
     #for word in final_response.choices[0].message.content.split():
     #     yield word + " "
     #     time.sleep(0.05)

##################
# Streamlit Code #
##################
st.title("PaulchWorks Agentic OpenAI Chatbot")
st.markdown("Welcome to the PaulchWorks Agentic OpenAI Chatbot. This chatbot is powered by OpenAI's engine. You can ask the chatbot any question and it will do its best to provide an answer.")
st.markdown("Write a message below to get started.")
st.sidebar.title("Traces...")
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
         st.markdown(message["content"])
if input := st.chat_input("Your message..."):
     st.session_state.messages.append({"role": "user", "content": input})
     with st.chat_message("user"):
         st.markdown(input)
     with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.write("Thinking...")
        response_text = response(input)
        message_placeholder.markdown(response_text)
        #response_text = st.write_stream(response(input))
        st.session_state.messages.append({"role": "assistant", "content": response_text})
