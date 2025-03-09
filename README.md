# Agentic OpenAI Chatbot
![image](https://github.com/user-attachments/assets/0179aa41-94d6-4475-b813-44275c7491e8)
![image](https://github.com/user-attachments/assets/8eaee6c4-379c-4751-8f1c-f8786e1d0d9d)
![image](https://github.com/user-attachments/assets/2519e5c7-9528-43a0-9ca1-df562dcd4a63)
## Overview
This chatbot leverages OpenAI's language models (GPT-4o-mini) alongside Azure Cognitive Search, Bing Web Search, and weather APIs to provide comprehensive answers. It combines internal document search, real-time web data, and weather information to address user queries.

---

## Key Features
1. **Multi-Source Information Retrieval**:
   - **Azure Cognitive Search**: Searches internal documents using hybrid semantic/vector search.
   - **Bing Web Search**: Fetches recent web results (past week) for up-to-date information.
   - **Weather API**: Provides current temperature data based on coordinates.

2. **Agentic Workflow**:
   - Automatically selects the appropriate tool (document/web/weather search) based on user input.
   - Uses OpenAI's model to generate responses augmented by tool outputs.

3. **Streamlit Interface**:
   - Simple web UI for user interaction with a trace sidebar showing tool execution details.

---

## Setup & Dependencies
1. **Environment Variables** (required in `.env`):
   ```ini
   OPENAI_API_KEY=your_openai_key
   AZURE_OPENAI_API_KEY=your_azure_openai_key
   AZURE_OPENAI_ENDPOINT=your_azure_endpoint
   AZURE_SEARCH_SERVICE_ENDPOINT=your_azure_search_endpoint
   AZURE_SEARCH_INDEX=your_search_index
   AZURE_SEARCH_API_KEY=your_azure_search_key
   AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG=your_semantic_config
   BING_ENDPOINT=your_bing_endpoint
   BING_KEY=your_bing_key
   ```

2. **Python Packages**:
   ```bash
   pip install openai azure-search-documents streamlit python-dotenv requests
   ```

---

## How It Works
1. **User Input**:
   - Submit a query via the Streamlit interface.

2. **Tool Selection**:
   - GPT-4o-mini determines which tool(s) to use (document search, web search, or weather lookup).

3. **Data Retrieval**:
   - **Document Search**: Generates embeddings for semantic/vector search in Azure.
   - **Web Search**: Queries Bing for recent results.
   - **Weather**: Fetches temperature data from Open-Meteo.

4. **Response Generation**:
   - Combines tool outputs with language model capabilities to form a natural language answer.

---

## Example Use Cases
- **Internal Knowledge**: "What's the company's 2023 revenue?" (uses Azure document search)
- **Current Events**: "Latest news on climate change?" (uses Bing web search)
- **Weather**: "What's the temperature in Paris?" (requires user-provided coordinates)

---

## Limitations
- Weather tool requires manual latitude/longitude input (not geocoded from location names).
- No error handling for invalid API credentials or network issues.
- Bing search results limited to 10 items and 1-week freshness.

---

## Running the App
```bash
streamlit run your_script_name.py
```

Access the interface at `http://localhost:8501`. Type queries in the message box and click "Send".
