import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn

class ResearchRequest(BaseModel):
    prompt: str

# Load environment variables from .env file
load_dotenv()

model = AzureChatOpenAI(azure_deployment=os.environ['AZURE_GPT_DEPLOYMENT'])
app = FastAPI(title="Research Assistant API")

async def process_prompt(prompt: str) -> Dict[str, Any]:
    mcp_client = MultiServerMCPClient({
        "arxiv": {
            "url": "http://arxiv-server:8000/sse",  # Use Docker service name
            "transport": "sse",
        },
        "docling": {
            "url": "http://docling-server:8001/sse",  # Use Docker service name
            "transport": "sse",
        }
    })
    async with mcp_client as client:
        agent = create_react_agent(model, client.get_tools())
        response = await agent.ainvoke({"messages": prompt}, debug=False)
        messages = [{i.type: i.content} for i in response['messages'] if i.content!='']
        return {"messages": messages}

@app.post("/research")
async def research(request: ResearchRequest):
    """Process a research query using arxiv and document analysis tools"""
    return await process_prompt(request.prompt)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)