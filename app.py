import os
import re
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = Flask(__name__)

print("Loading embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory="vector_db",
    embedding_function=embeddings
)

retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}
)

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

UIET_PAGES = {
    "admission":   "https://uiet.puchd.ac.in/?page_id=726",
    "courses":     "https://uiet.puchd.ac.in/?page_id=44",
    "fee":         "https://uiet.puchd.ac.in/?page_id=14671",
    "about":       "https://uiet.puchd.ac.in/?page_id=102",
    "be_admit":    "https://uiet.puchd.ac.in/?page_id=5555",
    "me_admit":    "https://uiet.puchd.ac.in/?page_id=105",
    "phd_admit":   "https://uiet.puchd.ac.in/?page_id=111",
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
            web_cache[url] = text[:6000]
            return web_cache[url]
    except Exception:
        pass
    return ""

print("Loading UIET website data...")
WEB_CONTEXT = ""
for name, url in UIET_PAGES.items():
    text = scrape_page(url)
    if text:
        WEB_CONTEXT += f"\n=== {name.upper()} ===\n{text}\n"
WEB_CONTEXT = WEB_CONTEXT[:8000]
print("Website loaded!")

session_memory = {}

# ─────────────────────────────────────────
# Complete hardcoded course list (authoritative)
# ─────────────────────────────────────────
UIET_COURSES = """
UIET Chandigarh offers the following programmes:

B.E. (Bachelor of Engineering) – 4 year undergraduate:
  1. Biotechnology
  2. Computer Science and Engineering
  3. Electrical and Electronics Engineering
  4. Electronics and Communication Engineering
  5. Information Technology
  6. Mechanical Engineering

M.E. / M.Tech (Postgraduate) – 2 year:
  1. M.E. Information Technology
  2. M.E. Computer Science and Engineering
  3. M.E. Cyber Security
  4. M.E. Electronics and Communication Engineering
  5. M.E. Biotechnology
  6. M.E. Electrical Engineering (Power System)
  7. M.E. Mechanical Engineering
  8. M.Tech. Micro Electronics

Ph.D. (Doctor of Philosophy):
  Available in all the above engineering disciplines and Applied Sciences.
  Admission through PUMEET / direct PhD entrance as per Panjab University norms.
"""

PROMPT_TEMPLATE = """You are Pravesh Saathi, a friendly voice assistant for UIET Chandigarh (Panjab University).

STRICT RULES:
1. Use the Authoritative Course List FIRST for any question about courses/programmes.
2. Use Local PDF Context for eligibility, documents, fee details, seats, process.
3. Use UIET Website Context for up-to-date dates, notices, and seat matrix.
4. ALWAYS give a direct, complete answer. NEVER say "please check the website".
5. Keep answers concise — max 6 sentences. This is a voice assistant.
6. Reply in plain simple English — NO markdown, NO bullet points, NO numbered lists, NO bold text.
7. For course lists: speak them naturally, separated by commas or "and".
8. Only answer about: courses, admission process, fees, eligibility, documents, seats, hostel, scholarships, counselling, PhD admission.
9. For out-of-scope topics say: "I can only help with admission related queries."
10. Use chat history for follow-up questions.

Authoritative Course List:
{course_list}

Local PDF Context:
{local_context}

UIET Website Context:
{web_context}

Chat History:
{chat_history}

Question: {question}

Answer (plain spoken English, no markdown, no bullet points):"""

OUT_OF_SCOPE = [
    "syllabus", "exam pattern", "question paper", "assignment",
    "project", "alumni", "professor", "faculty", "staff",
    "placement company", "timetable", "lecture",
    "attendance", "marksheet", "cgpa", "backlog"
]

def is_out_of_scope(query):
    return any(word in query.lower() for word in OUT_OF_SCOPE)

def format_history(history):
    formatted = ""
    for msg in history[-8:]:
        if isinstance(msg, HumanMessage):
            formatted += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            formatted += f"Assistant: {msg.content}\n"
    return formatted

def clean_for_voice(text: str) -> str:
    """
    Strip all markdown / formatting artifacts so TTS speaks naturally.
    This runs server-side before sending to the frontend.
    """
    # Remove bold (**text** or __text__)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    # Remove italic (*text* or _text_)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    # Remove inline code / backticks
    text = re.sub(r'`+', '', text)
    # Remove markdown headers
    text = re.sub(r'#{1,6}\s*', '', text)
    # Convert numbered list items → natural pause ("1. X" → "X")
    text = re.sub(r'^\s*\d+[\.\)]\s+', '', text, flags=re.MULTILINE)
    # Remove bullet/dash list markers
    text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)
    # Replace colons with period-space for natural pause
    text = re.sub(r':\s*\n', '. ', text)
    text = re.sub(r':\s+', '. ', text)
    # Em-dash / en-dash → comma
    text = re.sub(r'[—–]', ', ', text)
    # Remove brackets and parentheses content that's non-essential
    text = re.sub(r'\(([^)]{0,30})\)', r'\1', text)  # keep short parenthetical
    text = re.sub(r'\[[^\]]*\]', '', text)
    # Collapse multiple newlines to single space
    text = re.sub(r'\n+', ' ', text)
    # Degree abbreviations → spoken form
    text = re.sub(r'B\.E\.', 'B E', text)
    text = re.sub(r'M\.E\.', 'M E', text)
    text = re.sub(r'B\.Tech\.?', 'B Tech', text, flags=re.IGNORECASE)
    text = re.sub(r'M\.Tech\.?', 'M Tech', text, flags=re.IGNORECASE)
    text = re.sub(r'Ph\.D\.?', 'PhD', text, flags=re.IGNORECASE)
    text = re.sub(r'P\.U\.', 'Panjab University', text)
    text = re.sub(r'U\.I\.E\.T\.', 'UIET', text)
    # Rs. amount → spoken
    text = re.sub(r'Rs\.?\s?(\d[\d,]*)', r'\1 rupees', text, flags=re.IGNORECASE)
    # + sign
    text = re.sub(r'\+', ' plus ', text)
    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def ask_rag(query, session_id="default"):
    if not query.strip():
        return "Please ask a question."

    if is_out_of_scope(query):
        return "I can only help with admission related queries like courses, fees, eligibility, documents and counselling process."

    if session_id not in session_memory:
        session_memory[session_id] = []

    history = session_memory[session_id]
    docs = retriever.invoke(query)
    local_context = "\n\n".join([doc.page_content for doc in docs])

    prompt = PROMPT_TEMPLATE.format(
        course_list=UIET_COURSES,
        local_context=local_context,
        web_context=WEB_CONTEXT,
        chat_history=format_history(history),
        question=query
    )

    response = llm.invoke(prompt)
    answer = response.content

    # Clean before storing and returning
    answer = clean_for_voice(answer)

    history.append(HumanMessage(content=query))
    history.append(AIMessage(content=answer))

    if len(history) > 20:
        session_memory[session_id] = history[-20:]

    return answer

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    session_id = data.get("session_id", "default")

    if not user_message:
        return jsonify({"error": "Message is empty"}), 400

    try:
        reply = ask_rag(user_message, session_id)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)