
Real-tim





https://github.com/user-attachments/assets/317bf5fb-c4bf-4ad2-bd2f-18c864efccf9

e Voice Conversation: 
Features a full voice call mode with custom front-end voice activity detection (VAD), silence detection, and audio recording.

** Ultra-Low Latency STT & TTS:**

**Speech-to-Text:**
Powered by Groq's API running whisper-large-v3 for lightning-fast transcriptions.
**Text-to-Speech:**
Powered by edge-tts utilizing the realistic, Indian-accented neural female voice (en-IN-NeerjaNeural).
**Document Intelligence & RAG:**
Uses LangChain and a Chroma Vector Database powered by HuggingFace Embeddings (sentence-transformers/all-MiniLM-L6-v2) to query official admission brochures and FAQs.
**Live Website Scraping Context:**
Automatically scrapes live admission links, fee structures, notices, and updates from the official UIET Puchd website.
**Breathtaking Interactive UI:**
Neon-dark Glassmorphic UI dashboard with high-quality Outfit typography.
Interactive 3D Shimmer Orb with responsive eye tracking and speaking animations.
Live Equalizer wave visualizer syncing with listening, thinking, and speaking states.

**Architecture & Data Flow**
Here is how the RAG-based voice pipeline works:

Mermaid diagram
Directory Structure
text

<img width="611" height="247" alt="image" src="https://github.com/user-attachments/assets/cb1eb4c3-5b7c-42ba-84d6-e503c7dc901d" />


** Quick Start
1. Prerequisites**
Ensure you have Python 3.10+ installed on your system.

**2. Installation**
Clone the repository and navigate to the project directory:

git clone https://github.com/your-username/pravesh-saathi-ai.git
cd pravesh-saathi-ai
**Create a virtual environment and activate it:**
**Windows:**
python -m venv venv
.\venv\Scripts\activate

**Install the dependencies:**
pip install -r requirements.txt

**3. Environment Setup**
Create a .env file in the root folder and add your API keys:
env

GROQ_API_KEY=your_groq_api_key_here
ZINGARO_CALLER_ID=+91XXXXXXXXXX

**4. Build the RAG Vector DB**
Place your admission PDF brochures inside the pdfs/ directory, then run the database builder:

bash
python create_vector_db.py
**
5. Launch the Server**
Start the Flask application:
python app.py
Open your browser and navigate to http://127.0.0.1:5000 to experience Pravesh Saathi AI.

** Tech Stack & Integrations**
**Backend:** Flask, Python-Dotenv
**Frontend**: HTML5, Vanilla CSS, Custom Canvas 3D Orb, AudioContext API
**Large Language Model:** llama-3.3-70b-versatile via Groq
**Audio Models:** whisper-large-v3 via Groq, Microsoft edge-tts
**Vector Store:** Chroma database
**Orchestration: **LangChain, HuggingFace Embeddings

**Contributing**
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

**Fork the Project**
Create your Feature Branch (git checkout -b feature/AmazingFeature)
Commit your Changes (git commit -m 'Add some AmazingFeature')
Push to the Branch (git push origin feature/AmazingFeature)
Open a Pull Request
