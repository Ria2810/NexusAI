from typing import Dict, TypedDict, List
from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Main LLM used for classification
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    verbose=True,
    temperature=0.5,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

class State(TypedDict):
    query: str
    category: str
    response: str

def trim_conversation(prompt: List) -> List:
    """Trim conversation history to the last 10 messages."""
    max_messages = 10
    return trim_messages(
        prompt,
        max_tokens=max_messages,
        strategy="last",
        token_counter=len,
        start_on="human",
        include_system=True,
        allow_partial=False,
    )

# ----------------------------------------------------------------
# Single-turn agent classes (no input() loops).
# Each method processes exactly one query, returning a string.
# ----------------------------------------------------------------

class LearningResourceAgent:
    def __init__(self, prompt: List):
        self.model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        self.prompt = prompt
        self.tools = [DuckDuckGoSearchResults()]

    def tutorial_agent(self, user_input: str) -> str:
        """Handles single-turn tutorial creation."""
        agent = create_tool_calling_agent(self.model, self.tools, self.prompt)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)
        response = agent_executor.invoke({"input": user_input})
        result = str(response.get("output")).replace("```markdown", "").strip()
        return result

    def query_bot(self, user_input: str) -> str:
        """Handles single-turn Q&A within 'learning resource' context."""
        self.prompt.append(HumanMessage(content=user_input))
        self.prompt = trim_conversation(self.prompt)
        response = self.model.invoke(self.prompt)
        self.prompt.append(AIMessage(content=response.content))
        return response.content

class InterviewAgent:
    def __init__(self, prompt: List):
        self.model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        self.prompt = prompt
        self.tools = [DuckDuckGoSearchResults()]

    def interview_questions(self, user_input: str) -> str:
        """Handles single-turn question generation for interviews."""
        agent = create_tool_calling_agent(self.model, self.tools, self.prompt)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True, handle_parsing_errors=True)
        response = agent_executor.invoke({"input": user_input, "chat_history": []})
        return str(response.get("output")).replace("```markdown", "").strip()

    def mock_interview(self, user_input: str) -> str:
        """Handles single-turn in a mock interview flow."""
        # Start with an initial interviewer message
        initial_message = "I am ready for the interview."
        self.prompt.append(HumanMessage(content=initial_message))
        self.prompt = trim_conversation(self.prompt)
        response = self.model.invoke(self.prompt)
        self.prompt.append(AIMessage(content=response.content))
        # Now handle the candidate's single message
        self.prompt.append(HumanMessage(content=user_input))
        return f"Interviewer: {response.content}\nCandidate: {user_input}"

class ResumeMaker:
    def __init__(self, prompt: List):
        self.model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        self.prompt = prompt
        self.tools = [DuckDuckGoSearchResults()]
        self.agent = create_tool_calling_agent(self.model, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True, handle_parsing_errors=True)

    def create_resume(self, user_input: str) -> str:
        """Handles single-turn resume creation or refinement."""
        response = self.agent_executor.invoke({"input": user_input, "chat_history": []})
        return str(response.get("output")).replace("```markdown", "").strip()

class JobSearch:
    def __init__(self, prompt: ChatPromptTemplate):
        self.model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        self.prompt = prompt
        self.tools = DuckDuckGoSearchResults()

    def find_jobs(self, user_input: str) -> str:
        results = self.tools.invoke(user_input)
        chain = self.prompt | self.model
        jobs = chain.invoke({"result": results}).content
        return str(jobs).replace("```markdown", "").strip()

# ----------------------------------------------------------------
# Classification / Routing logic
# ----------------------------------------------------------------

def categorize(state: State) -> State:
    """Top-level classification: choose 1 of 4 categories."""
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following customer query into one of these categories:\n"
        "1: Learn Generative AI Technology\n"
        "2: Resume Making\n"
        "3: Interview Preparation\n"
        "4: Job Search\n"
        "Give the number only as an output.\n\n"
        "Examples:\n"
        "1. Query: 'What are the basics of generative AI, and how can I start learning it?' -> 1\n"
        "2. Query: 'Can you help me improve my resume for a tech position?' -> 2\n"
        "3. Query: 'What are some common questions asked in AI interviews?' -> 3\n"
        "4. Query: 'Are there any job openings for AI engineers?' -> 4\n\n"
        "Now, categorize the following customer query:\n"
        "Query: {query}"
    )
    chain = prompt | llm
    category = chain.invoke({"query": state["query"]}).content
    return {"category": category}

def handle_learning_resource(state: State) -> State:
    """Second-level classification for learning resource queries: 'tutorial' or 'question'."""
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following user query into one of these categories:\n\n"
        "Categories:\n"
        "- Tutorial: For queries related to creating tutorials, blogs, or documentation on generative AI.\n"
        "- Question: For general queries asking about generative AI topics.\n"
        "- Default to Question if the query doesn't fit either of these categories.\n\n"
        "Now, categorize the following user query:\n"
        "The user query is: {query}\n"
    )
    chain = prompt | llm
    response = chain.invoke({"query": state["query"]}).content
    return {"category": response}

def handle_interview_preparation(state: State) -> State:
    """Second-level classification for interview queries: 'mock' or 'question'."""
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following user query into one of these categories:\n\n"
        "Categories:\n"
        "- Mock: For requests related to mock interviews.\n"
        "- Question: For general queries asking about interview topics or preparation.\n"
        "Default to Question if the query doesn't fit either.\n\n"
        "Now, categorize the following user query:\n"
        "The user query is: {query}\n"
    )
    chain = prompt | llm
    response = chain.invoke({"query": state["query"]}).content
    return {"category": response}

def job_search(state: State) -> State:
    """Single-turn job search route."""
    prompt = ChatPromptTemplate.from_template(
        '''Your task is to refactor and return the output for the following content which includes
        the jobs available in the market. Refactor such that the user can refer easily. Content: {result}'''
    )
    job_search_agent = JobSearch(prompt)
    # Single turn
    result = job_search_agent.find_jobs(state["query"])
    return {"response": result}

def handle_resume_making(state: State) -> State:
    """Single-turn resume making route."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", '''You are a skilled resume expert with extensive experience in crafting resumes tailored for tech roles, especially in AI and Generative AI. 
        Your task is to create or refine a resume template for an AI Engineer specializing in Generative AI, incorporating trending keywords and technologies 
        in the current job market. You may ask for additional details if needed. The final output should be in markdown format.'''),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    resume_maker = ResumeMaker(prompt)
    result = resume_maker.create_resume(state["query"])
    return {"response": result}

def tutorial_agent(state: State) -> State:
    """Single-turn tutorial creation route."""
    system_message = (
        "You are a knowledgeable assistant specializing as a Senior Generative AI Developer with extensive experience in both development and tutoring. "
        "Additionally, you are an experienced blogger who creates tutorials focused on Generative AI. "
        "Your task is to develop high-quality tutorial blogs in markdown with coding examples based on the user's requirements."
    )
    prompt = [
        SystemMessage(content=system_message)
    ]
    agent = LearningResourceAgent(prompt)
    result = agent.tutorial_agent(state["query"])
    return {"response": result}

def ask_query_bot(state: State) -> State:
    """Single-turn Q&A route within 'learning resource' category."""
    system_message = (
        "You are an expert Generative AI Engineer with extensive experience in training and guiding others in AI engineering. "
        "You have a strong track record of solving complex problems and addressing various challenges in AI. "
        "Your role is to assist users by providing insightful solutions and expert advice on their queries."
    )
    prompt = [SystemMessage(content=system_message)]
    agent = LearningResourceAgent(prompt)
    result = agent.query_bot(state["query"])
    return {"response": result}

def interview_topics_questions(state: State) -> State:
    """Single-turn route for listing or discussing interview questions."""
    system_message = (
        "You are a good researcher in finding interview questions for Generative AI topics and jobs. "
        "Provide top questions with references and links if possible. The output should be in markdown format."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = InterviewAgent(prompt)
    result = agent.interview_questions(state["query"])
    return {"response": result}

def mock_interview(state: State) -> State:
    """Single-turn route for mock interview conversation."""
    system_message = (
        "You are a Generative AI Interviewer with extensive experience conducting interviews for AI roles. "
        "Engage in a brief mock interview. Provide the interviewer prompt and then handle the candidate's single response."
    )
    prompt = [SystemMessage(content=system_message)]
    agent = InterviewAgent(prompt)
    result = agent.mock_interview(state["query"])
    return {"response": result}

# ----------------------------------------------------------------
# Graph routing
# ----------------------------------------------------------------

def route_query(state: State):
    """Top-level route: 1..4."""
    category = state["category"]
    if "1" in category:
        return "handle_learning_resource"
    elif "2" in category:
        return "handle_resume_making"
    elif "3" in category:
        return "handle_interview_preparation"
    elif "4" in category:
        return "job_search"
    else:
        return False  # unrecognized

def route_learning(state: State):
    """Sub-route for learning resource: 'tutorial' or 'question'."""
    category = state["category"].lower()
    if "tutorial" in category:
        return "tutorial_agent"
    elif "question" in category:
        return "ask_query_bot"
    else:
        return "ask_query_bot"  # default

def route_interview(state: State):
    """Sub-route for interview: 'mock' or 'question'."""
    category = state["category"].lower()
    if "mock" in category:
        return "mock_interview"
    elif "question" in category:
        return "interview_topics_questions"
    else:
        return "interview_topics_questions"  # default

# Build the graph
workflow = StateGraph(State)
workflow.add_node("categorize", categorize)
workflow.add_node("handle_learning_resource", handle_learning_resource)
workflow.add_node("handle_resume_making", handle_resume_making)
workflow.add_node("handle_interview_preparation", handle_interview_preparation)
workflow.add_node("job_search", job_search)
workflow.add_node("tutorial_agent", tutorial_agent)
workflow.add_node("ask_query_bot", ask_query_bot)
workflow.add_node("mock_interview", mock_interview)
workflow.add_node("interview_topics_questions", interview_topics_questions)

workflow.add_edge(START, "categorize")
workflow.add_conditional_edges(
    "categorize",
    route_query,
    {
        "handle_learning_resource": "handle_learning_resource",
        "handle_resume_making": "handle_resume_making",
        "handle_interview_preparation": "handle_interview_preparation",
        "job_search": "job_search"
    }
)
workflow.add_conditional_edges(
    "handle_learning_resource",
    route_learning,
    {
        "tutorial_agent": "tutorial_agent",
        "ask_query_bot": "ask_query_bot"
    }
)
workflow.add_conditional_edges(
    "handle_interview_preparation",
    route_interview,
    {
        "mock_interview": "mock_interview",
        "interview_topics_questions": "interview_topics_questions"
    }
)

# Terminal edges
workflow.add_edge("handle_resume_making", END)
workflow.add_edge("job_search", END)
workflow.add_edge("tutorial_agent", END)
workflow.add_edge("ask_query_bot", END)
workflow.add_edge("mock_interview", END)
workflow.add_edge("interview_topics_questions", END)

workflow.set_entry_point("categorize")
app = workflow.compile()

def run_user_query(query: str) -> Dict[str, str]:
    """
    Process a single user query from scratch through the workflow.
    Returns the final category (e.g. "handle_resume_making") 
    and the single-turn response.
    """
    results = app.invoke({"query": query})
    return {
        "category": results.get("category", ""),
        "response": results.get("response", "")
    }
