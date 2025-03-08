# Description: This file contains the main code for the Agentic OpenAI Chatbot. The chatbot is powered by OpenAI's engine and uses Azure Cognitive Search to search for documents and Bing Web Search to search the web for information.
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
     completion = client.chat.completions.create(
          model="gpt-4o-mini",
          messages=[
               {
                    "role": "developer", 
                    "content": "You are a helpful assistant. The current date is " + current_date
                    },
                {
                    "role": "user", 
                    "content": input
                    }
            ],
          tools=tools,
          tool_choice="auto"
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
     
     st.sidebar.write("Function currently being executed: " + tool_call_name, args)
     
     messages = [{
            "role": "user",
            "content": input
     }]
     messages.append({
          "role": "developer", 
          "content": "You are a helpful assistant. The current date is " + current_date
    })
     messages.append({
            "role": "assistant",
            "tool_calls": [tool_call.model_dump()]
     })
     messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(results)
     })
     completion_2 = client.chat.completions.create(
           model="gpt-4o-mini",
           messages=messages,
           tools=tools,
     )
     return completion_2.choices[0].message.content

##################
# Streamlit Code #
##################
st.title("Agentic OpenAI Chatbot")
st.markdown("Welcome to the Agentic OpenAI Chatbot. This chatbot is powered by OpenAI's engine. You can ask the chatbot any question and it will do its best to provide an answer.")
st.markdown("Write a message below to get started.")
st.sidebar.title("Traces...")
with st.form("form"):
    input = st.text_area("My question is...")
    submit = st.form_submit_button("Over to you, Agentic!")
    if submit:
        st.write(response(input))
