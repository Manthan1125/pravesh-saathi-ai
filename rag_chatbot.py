import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory="vector_db",
    embedding_function=embeddings
)

print("Total Chunks:", vector_db._collection.count())

retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 15}
)

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# UIET pages
UIET_PAGES = {
    "admission": "https://uiet.puchd.ac.in/?page_id=726",
    "courses":   "https://uiet.puchd.ac.in/?page_id=44",
    "fee":       "https://uiet.puchd.ac.in/?page_id=14671",
    "about":     "https://uiet.puchd.ac.in/?page_id=102",
}

web_cache = {}

def scrape_page(url):
    if url in web_cache:
        return web_cache[url]
    try:
        r = requests.get(url, timeout=8)
        soup = BeautifulSoup(r.text, "html.parser")
        content = soup.find("div", class_="entry-content")
        if content:
            lines = [l.strip() for l in content.get_text(separator="\n").splitlines()]
            text = "\n".join(l for l in lines if l)
            web_cache[url] = text[:5000]
            return web_cache[url]
    except:
        pass
    return ""

def get_all_web_context():  
    combined = ""
    for name, url in UIET_PAGES.items():
        text = scrape_page(url)
        if text:
            combined += f"\n=== {name.upper()} ===\n{text}\n"
    return combined[:8000]

print("Fetching UIET website data...")
WEB_CONTEXT = get_all_web_context()
print("Website data loaded!")

PROMPT_TEMPLATE = """You are Pravesh Saathi, a voice assistant for UIET Chandigarh (Panjab University).

STRICT RULES:
1. You have TWO sources: Local PDFs and UIET Website. Use BOTH to give the most accurate answer.
2. Website data is more up-to-date — prefer it for courses, seats, fees, dates.
3. PDF data is more detailed — prefer it for eligibility, documents, process.
4. ALWAYS give a direct, complete answer. NEVER say "please check the website".
5. For course lists — give the COMPLETE list from both sources combined.
6. Keep answers concise — max 8 lines. This is a voice agent.
7. Reply in simple English.
8. Only answer about: courses, admission process, fees, eligibility, documents, seats, hostel, scholarships, counselling, important dates.
9. For out-of-scope topics (syllabus, exams, professors, projects, alumni, placement) say: "I can only help with admission-related queries."

Local PDF Context:
{local_context}

UIET Website Context:
{web_context}

Chat History:
{chat_history}

Question: {question}

Answer:"""

OUT_OF_SCOPE = [
    "syllabus", "exam pattern", "question paper", "assignment",
    "project", "alumni", "professor", "faculty", "staff",
    "placement company", "timetable", "schedule", "lecture",
    "attendance", "result", "marksheet", "cgpa", "backlog"
]

chat_history = []

def is_out_of_scope(query):
    q = query.lower()
    return any(word in q for word in OUT_OF_SCOPE)

def format_history(history):
    formatted = ""
    for msg in history[-6:]:
        if isinstance(msg, HumanMessage):
            formatted += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            formatted += f"Assistant: {msg.content}\n"
    return formatted

def ask(query):
    if not query.strip():
        return None

    if is_out_of_scope(query):
        return "I can only help with admission-related queries like courses, fees, eligibility, documents and counselling process."

    # Hamesha dono sources use karo
    docs = retriever.invoke(query)
    local_context = "\n\n".join([doc.page_content for doc in docs])

    prompt = PROMPT_TEMPLATE.format(
        local_context=local_context,
        web_context=WEB_CONTEXT,
        chat_history=format_history(chat_history),
        question=query
    )

    response = llm.invoke(prompt)
    answer = response.content

    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=answer))
    return answer

print("\n🤖 UIET Chandigarh AI Assistant Ready")
print("--------------------------------------")
print("Type 'exit' to stop\n")

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    if not query.strip():
        continue
    answer = ask(query)
    if answer:
        print(f"\nPravesh Saathi: {answer}\n")