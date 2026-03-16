import os
import re
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

load_dotenv()

# Shared state passed between all agents
class ResearchState(TypedDict):
    topic: str
    questions: List[str]
    research: List[str]
    report: str
    critic_feedback: str

# Initialize LLM and search tool
llm = ChatGroq(model="llama-3.3-70b-versatile")
search = TavilySearch(max_results=3)


# Node 1: Planner - breaks topic into research questions
def planner(state: ResearchState) -> dict:
    print("\n🧠 Planner is thinking...\n")

    prompt = f"""You are a research planner. Given a topic, generate exactly 5 specific research questions that would help someone understand it deeply.

Topic: {state["topic"]}

Return ONLY a numbered list like this:
1. First question
2. Second question
3. Third question
4. Fourth question
5. Fifth question

No extra text, just the 5 questions."""

    response = llm.invoke(prompt)

    # Parse the numbered list into a Python list
    lines = response.content.strip().split("\n")
    questions = []
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit():
            question = line.split(". ", 1)[-1]
            questions.append(question)

    print(f"Questions generated: {questions}\n")
    return {"questions": questions}


# Node 2: Researcher - searches the web for each question
def researcher(state: ResearchState) -> dict:
    print("🔍 Researcher is searching...\n")

    all_research = []

    for question in state["questions"]:
        print(f"  Searching: {question[:60]}...")
        try:
            results = search.invoke(question)
            combined = f"Question: {question}\n"
            for r in results.get("results", []):
                combined += f"Source: {r['url']}\n{r['content'][:300]}\n\n"
        except Exception as e:
            print(f"  ⚠️  Search failed for '{question[:50]}...': {e}")
            combined = f"Question: {question}\n[Search failed: {e}]\n"

        all_research.append(combined)

    print(f"\n✅ Research complete — {len(all_research)} topics covered\n")
    return {"research": all_research}


# Node 3: Synthesizer - combines research into a first-draft report
def synthesizer(state: ResearchState) -> dict:
    print("✍️  Synthesizer is writing the report...\n")

    all_research = "\n\n".join(state["research"])

    prompt = f"""You are a research synthesizer. Using the research below, write a clear and well-structured report on the topic.

Topic: {state["topic"]}

Research:
{all_research}

Write a 3-4 paragraph report that:
- Summarizes the key findings
- Highlights important insights
- Notes any disagreements or gaps in the research
- Ends with a conclusion

Write the report now:"""

    response = llm.invoke(prompt)
    return {"report": response.content.strip()}


# Node 4: Critic - reviews the report and identifies weaknesses
def critic(state: ResearchState) -> dict:
    print("🔎 Critic is reviewing the report...\n")

    prompt = f"""You are a critical research reviewer. Read the report below and identify exactly 2-3 specific weak points or gaps.

Topic: {state["topic"]}

Report:
{state["report"]}

Return ONLY a numbered list of 2-3 weak points or gaps, like this:
1. [Specific weakness or missing information]
2. [Specific weakness or missing information]
3. [Specific weakness or missing information]

No extra text, just the numbered list."""

    response = llm.invoke(prompt)
    feedback = response.content.strip()
    print(f"Critic feedback:\n{feedback}\n")
    return {"critic_feedback": feedback}


# Node 5: Rewriter - improves the report using critic feedback
def rewriter(state: ResearchState) -> dict:
    print("✨ Rewriter is improving the report...\n")

    all_research = "\n\n".join(state["research"])

    prompt = f"""You are a research synthesizer revising a report based on critical feedback.

Topic: {state["topic"]}

Original Report:
{state["report"]}

Critic Feedback (weaknesses to address):
{state["critic_feedback"]}

Research (for reference):
{all_research}

Rewrite the report as 3-4 paragraphs, directly addressing each piece of critic feedback while preserving the strong parts of the original. End with a conclusion.

Write the improved report now:"""

    response = llm.invoke(prompt)
    return {"report": response.content.strip()}


def topic_slug(topic: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:60]


def run_research(app, topic: str) -> None:
    result = app.invoke({
        "topic": topic,
        "questions": [],
        "research": [],
        "report": "",
        "critic_feedback": "",
    })

    print("\n📄 FINAL REPORT\n")
    print("=" * 50)
    print(result["report"])
    print("=" * 50 + "\n")

    os.makedirs("reports", exist_ok=True)
    filename = f"reports/{topic_slug(topic)}.md"
    with open(filename, "w") as f:
        f.write(f"# {topic}\n\n{result['report']}\n")
    print(f"📁 Report saved to {filename}\n")


if __name__ == "__main__":
    # Build the graph
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner)
    graph.add_node("researcher", researcher)
    graph.add_node("synthesizer", synthesizer)
    graph.add_node("critic", critic)
    graph.add_node("rewriter", rewriter)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "synthesizer")
    graph.add_edge("synthesizer", "critic")
    graph.add_edge("critic", "rewriter")
    graph.add_edge("rewriter", END)

    app = graph.compile()

    print("🔬 Multi-Agent Research System")

    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
        run_research(app, topic)
    else:
        print("Type 'quit' to exit\n")
        while True:
            topic = input("Enter a research topic: ")
            if topic.lower() == "quit":
                break
            run_research(app, topic)
