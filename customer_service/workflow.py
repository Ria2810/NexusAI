from typing import TypedDict, Dict
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Import the workflow graph framework from LangGraph (assumed to be installed)
from langgraph.graph import StateGraph, END  # Ensure you have this library or equivalent
from IPython.display import display, Image
from langchain_core.runnables.graph import MermaidDrawMethod
from dotenv import load_dotenv
import os

# Load environment variables and set OpenAI API key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Define the state for our workflow
class State(TypedDict):
    query: str
    category: str
    sentiment: str
    response: str

def categorize(state: State) -> State:
    """
    Categorize the customer query into Technical, Billing, or General.
    Personalized for TechNova Solutions.
    """
    prompt = ChatPromptTemplate.from_template(
        "At TechNova Solutions, we strive to provide quick and accurate support. "
        "Please categorize the following customer query into one of these categories: "
        "Technical, Billing, or General. Query: {query}"
    )
    chain = prompt | ChatOpenAI(temperature=0)
    category = chain.invoke({"query": state["query"]}).content
    return {"category": category}

def analyze_sentiment(state: State) -> State:
    """
    Analyze the sentiment of the customer query as Positive, Neutral, or Negative.
    Personalized for TechNova Solutions.
    """
    prompt = ChatPromptTemplate.from_template(
        "At TechNova Solutions, your experience matters to us. Analyze the sentiment of the "
        "following customer query and respond with either 'Positive', 'Neutral', or 'Negative'. "
        "Query: {query}"
    )
    chain = prompt | ChatOpenAI(temperature=0)
    sentiment = chain.invoke({"query": state["query"]}).content
    return {"sentiment": sentiment}

def handle_technical(state: State) -> State:
    """
    Provide a technical support response to the query.
    """
    prompt = ChatPromptTemplate.from_template(
        "At TechNova Solutions, our technical support team is here to help. Provide a detailed "
        "technical support response to the following query: {query}"
    )
    chain = prompt | ChatOpenAI(temperature=0)
    response = chain.invoke({"query": state["query"]}).content
    return {"response": response}

def handle_billing(state: State) -> State:
    """
    Provide a billing support response to the query.
    """
    prompt = ChatPromptTemplate.from_template(
        "At TechNova Solutions, our billing support team is ready to assist. Provide a clear billing "
        "support response to the following query: {query}"
    )
    chain = prompt | ChatOpenAI(temperature=0)
    response = chain.invoke({"query": state["query"]}).content
    return {"response": response}

def handle_general(state: State) -> State:
    """
    Provide a general support response to the query.
    """
    prompt = ChatPromptTemplate.from_template(
        "At TechNova Solutions, we value all customer inquiries. Provide a friendly general support "
        "response to the following query: {query}"
    )
    chain = prompt | ChatOpenAI(temperature=0)
    response = chain.invoke({"query": state["query"]}).content
    return {"response": response}

def escalate(state: State) -> State:
    """
    Escalate the query to a human agent due to negative sentiment.
    """
    return {"response": "Your query has been escalated to a human agent at TechNova Solutions due to its negative sentiment. Please hold while we connect you."}

def route_query(state: State) -> str:
    """
    Route the query based on its sentiment and category.
    """
    if state["sentiment"].strip().lower() == "negative":
        return "escalate"
    elif state["category"].strip().lower() == "technical":
        return "handle_technical"
    elif state["category"].strip().lower() == "billing":
        return "handle_billing"
    else:
        return "handle_general"

# Create the graph
workflow = StateGraph(State)

# Add nodes to the graph
workflow.add_node("categorize", categorize)
workflow.add_node("analyze_sentiment", analyze_sentiment)
workflow.add_node("handle_technical", handle_technical)
workflow.add_node("handle_billing", handle_billing)
workflow.add_node("handle_general", handle_general)
workflow.add_node("escalate", escalate)

# Add edges between nodes
workflow.add_edge("categorize", "analyze_sentiment")
workflow.add_conditional_edges(
    "analyze_sentiment",
    route_query,
    {
        "handle_technical": "handle_technical",
        "handle_billing": "handle_billing",
        "handle_general": "handle_general",
        "escalate": "escalate"
    }
)
workflow.add_edge("handle_technical", END)
workflow.add_edge("handle_billing", END)
workflow.add_edge("handle_general", END)
workflow.add_edge("escalate", END)

# Set the entry point of the graph
workflow.set_entry_point("categorize")

# Compile the graph into an executable app
app = workflow.compile()

def run_customer_support(query: str) -> Dict[str, str]:
    """
    Process a customer query through the LangGraph workflow.
    
    Args:
        query (str): The customer's query
        
    Returns:
        Dict[str, str]: A dictionary containing the query's category, sentiment, and response
    """
    results = app.invoke({"query": query})
    return {
        "category": results["category"],
        "sentiment": results["sentiment"],
        "response": results["response"]
    }

# Example usage (uncomment to test locally):
# query = "My internet connection keeps dropping. Can you help?"
# result = run_customer_support(query)
# print(f"Query: {query}")
# print(f"Category: {result['category']}")
# print(f"Sentiment: {result['sentiment']}")
# print(f"Response: {result['response']}")
