from dotenv import load_dotenv
from langchain_ollama import ChatOllama
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

# Initialize LLM and search tool
llm = ChatOllama(model="llama3.2")
search = TavilySearch(max_results=2)

# Node 1: Planner - breaks topic into research questions
def planner(state: ResearchState) -> dict:
    print("\n🧠 Planner is thinking...\n")
    
    prompt = f"""You are a research planner. Given a topic, generate exactly 3 specific research questions that would help someone understand it deeply.

Topic: {state["topic"]}

Return ONLY a numbered list like this:
1. First question
2. Second question  
3. Third question

No extra text, just the 3 questions."""

    response = llm.invoke(prompt)
    
    # Parse the numbered list into a Python list
    lines = response.content.strip().split("\n")
    questions = []
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit():
            # Remove the number and period at the start (e.g. "1. ")
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
        results = search.invoke(question)
        
        # Combine the search results into one text block
        combined = f"Question: {question}\n"
        for r in results.get("results", []):
            combined += f"Source: {r['url']}\n{r['content'][:300]}\n\n"
        
        all_research.append(combined)
    
    print(f"\n✅ Research complete — {len(all_research)} topics covered\n")
    return {"research": all_research}


# Node 3: Synthesizer - combines research into a final report
def synthesizer(state: ResearchState) -> dict:
    print("✍️  Synthesizer is writing the report...\n")
    
    # Join all research into one big context block
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
    report = response.content.strip()
    
    return {"report": report}

if __name__ == "__main__":
    # Build the graph
    graph = StateGraph(ResearchState)
    
    graph.add_node("planner", planner)
    graph.add_node("researcher", researcher)
    graph.add_node("synthesizer", synthesizer)
    
    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "synthesizer")
    graph.add_edge("synthesizer", END)
    
    app = graph.compile()
    
    print("🔬 Multi-Agent Research System")
    print("Type 'quit' to exit\n")
    
    while True:
        topic = input("Enter a research topic: ")
        if topic.lower() == "quit":
            break
        
        result = app.invoke({
            "topic": topic,
            "questions": [],
            "research": [],
            "report": ""
        })
        
        print("\n📄 FINAL REPORT\n")
        print("=" * 50)
        print(result["report"])
        print("=" * 50 + "\n")