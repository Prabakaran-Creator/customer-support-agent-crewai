import os
from datetime import datetime

import gspread
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai.tools import tool
from crewai_tools import SerperDevTool

load_dotenv()

SUPPORT_TEAM_REPLY = (
    "Thank you for your question. I have passed it to our support team, "
    "and they will get back to you within 24 hours."
)


# Tool 1 (RAG): reads our FAQ file and gives it to the agent
@tool("FAQ Search")
def faq_search(query: str) -> str:
    """Search the company FAQ and return its content for the given question."""
    with open("knowledge/faq.txt", "r", encoding="utf-8") as file:
        return file.read()


# Tool 2: adds one row to our Google Sheet
@tool("Google Sheet Logger")
def sheet_logger(question: str, answer: str, source: str) -> str:
    """Add one row to the Support Log Google Sheet.
    source must be one of: FAQ, Web, Support Team."""
    client = gspread.service_account(filename="credentials.json")
    sheet = client.open_by_key(os.getenv("GOOGLE_SHEET_ID")).sheet1
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sheet.append_row([now, question, answer, source])
    return "Saved to Google Sheet"


# Agent 1: Support Agent
support_agent = Agent(
    role="Customer Support Agent",
    goal="Answer customer questions using only the company FAQ.",
    backstory="You are a friendly and polite support agent for BrightMart Online Store.",
    tools=[faq_search],
    llm="gpt-4o-mini",
    verbose=False,
)

# Task for Agent 1
answer_task = Task(
    description=(
        "Customer question: {question}\n"
        "Use the FAQ Search tool and answer in a short, friendly way. "
        "If the FAQ does not contain the answer, reply with exactly: NOT_FOUND"
    ),
    expected_output="A short, friendly answer, or NOT_FOUND.",
    agent=support_agent,
)

# Agent 2: Research Agent (web search)
research_agent = Agent(
    role="Research Agent",
    goal="Find the right answer on the web when the FAQ has no answer, without inventing facts.",
    backstory=(
        "You are a careful researcher who writes simple, customer-friendly answers. "
        "You never make promises about company policies, prices or shipping "
        "unless a web search result clearly confirms them."
    ),
    tools=[SerperDevTool()],
    llm="gpt-4o-mini",
    verbose=False,
)

# Task for Agent 2
research_task = Task(
    description=(
        "Customer question: {question}\n"
        "Look at the previous answer.\n"
        "1. If it is a real answer (not NOT_FOUND), repeat it exactly and do NOT search.\n"
        "2. If it is NOT_FOUND and the question is about BrightMart itself "
        "(its stores, branches, shipping, delivery areas, orders, prices, offers or policies), "
        "do NOT search. Reply exactly: '" + SUPPORT_TEAM_REPLY + "'\n"
        "3. Otherwise it is a general question. You MUST use the search tool first, "
        "then write a short, friendly answer based only on what the search results say.\n"
        "Write plain text only. Do not use markdown, links, URLs or bullet points. "
        "Keep the answer to 2 or 3 short sentences."
    ),
    expected_output="A short, friendly, plain-text answer for the customer.",
    agent=research_agent,
    context=[answer_task],
)

# Agent 3: Record Keeper (Google Sheet entry)
sheet_agent = Agent(
    role="Record Keeper",
    goal="Save every customer question and answer in the Google Sheet.",
    backstory="You are an organised assistant who keeps accurate support records.",
    tools=[sheet_logger],
    llm="gpt-4o-mini",
    verbose=False,
)

# Task for Agent 3 (the source is decided by our code, not guessed)
sheet_task = Task(
    description=(
        "Save this support record.\n"
        "Question: {question}\n"
        "Answer: {answer}\n"
        "Source: {source}\n"
        "Use the Google Sheet Logger tool exactly once with these exact values, "
        "without changing any word. After saving, reply with ONLY the answer text."
    ),
    expected_output="The answer text, nothing else.",
    agent=sheet_agent,
)

# Crew 1: Agent 1 and Agent 2 find the answer
crew_answer = Crew(
    agents=[support_agent, research_agent],
    tasks=[answer_task, research_task],
)

# Crew 2: Agent 3 saves the record
crew_log = Crew(
    agents=[sheet_agent],
    tasks=[sheet_task],
)

# Live chat loop
print("Welcome to BrightMart Support! Type your question (or type exit to stop).")

while True:
    question = input("\nYou: ").strip()

    if question.lower() == "exit":
        print("Thank you for contacting BrightMart. Have a great day!")
        break

    if question == "":
        continue

    print("Support: One moment please...")

    # Step 1 and 2: find the answer
    result = crew_answer.kickoff(inputs={"question": question})
    final_answer = str(result).strip()
    faq_result = str(answer_task.output.raw).strip()

    # Decide the source with simple rules
    if "NOT_FOUND" not in faq_result:
        source = "FAQ"
    elif "passed it to our support team" in final_answer:
        source = "Support Team"
    else:
        source = "Web"

    # Step 3: save the record in Google Sheet
    crew_log.kickoff(
        inputs={"question": question, "answer": final_answer, "source": source}
    )

    print("\nSupport:", final_answer)