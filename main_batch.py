import os
import asyncio
from datetime import datetime

import gspread
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai.tools import tool
from crewai_tools import SerperDevTool

load_dotenv()

# The customers' questions (all of them run at the same time)
questions = [
    "How long does a refund take?",
    "Is UPI payment safe?",
    "Do you ship to Dubai?",
]


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
        "2. If it is NOT_FOUND, you MUST use the search tool first, then write a short, "
        "friendly answer based only on what the search results say.\n"
        "3. If the question is about BrightMart's own policy, shipping, prices or orders "
        "and the web cannot confirm it, do not guess. Reply exactly: "
        "'Thank you for your question. I have passed it to our support team, "
        "and they will get back to you within 24 hours.'\n"
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

# Task for Agent 3
sheet_task = Task(
    description=(
        "Customer question: {question}\n"
        "Take the final answer from the research task. "
        "Decide the source using the FIRST task's result (the FAQ answer):\n"
        "- If the first task's result is NOT_FOUND and the final answer says it was "
        "passed to our support team, the source is 'Support Team'.\n"
        "- If the first task's result is NOT_FOUND and the final answer is a real answer, "
        "the source is 'Web'.\n"
        "- Only if the first task's result is a real answer (not NOT_FOUND), the source is 'FAQ'.\n"
        "Use the Google Sheet Logger tool exactly once to save the question, "
        "the final answer, and the source. "
        "After saving, reply with ONLY the final answer text for the customer."
    ),
    expected_output="The final customer-friendly answer, nothing else.",
    agent=sheet_agent,
    context=[answer_task, research_task],
)

# The crew: 3 agents working in order for each question
crew = Crew(
    agents=[support_agent, research_agent, sheet_agent],
    tasks=[answer_task, research_task, sheet_task],
)


# Run all questions at the same time (parallel)
async def main():
    inputs = [{"question": q} for q in questions]
    return await crew.kickoff_for_each_async(inputs=inputs)


results = asyncio.run(main())

# Show the customer-friendly answers
for q, r in zip(questions, results):
    print("\nCustomer:", q)
    print("Reply   :", r)