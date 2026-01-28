# AI Powered Lecture Voice-to-Notes Generator | Professional Neural Transcript & Analytics Tool

A high-performance, AI dashboard built to transform lecture audio and YouTube videos into structured academic intelligence. This application runs entirely on your local hardware using state-of-the-art Open Source models, ensuring data privacy and zero API costs.

##  Key Features
- **Dual-Source Input:** Process local `.mp3`/`.wav` files or direct YouTube URLs.
- **Speech-to-Text:** Powered by OpenAI's **Whisper (Base)** for high-accuracy transcription.
- **Neural Summarization:** Utilizes **Facebook's BART-Large-CNN** for executive-level summaries.
- **Semantic Concept Extraction:** Automatically identifies key terminology and definitions using **spaCy**.
- **Analytics Engine:** Calculates word counts and estimated reading times for study planning.
- **Arctic Professional UI:** A high-contrast, light-themed dashboard designed for clarity and focus.

##  Architecture & Tech Stack
- **Frontend:** Streamlit (Python-based Web Framework)
- **Audio Processing:** yt-dlp (YouTube Extraction) & FFmpeg
- **Machine Learning Models:**
  - `openai-whisper` (Transcription)
  - `transformers` (BART-Large-CNN for Summarization)
  - `spacy` (Natural Language Processing / Entity Extraction)
- **Backend:** PyTorch (Neural Engine)



##  Installation & Setup

### Prerequisites
- **Python 3.12.8**
- **FFmpeg:** Required for audio processing.
  - *Windows:* `choco install ffmpeg`
  - *Mac:* `brew install ffmpeg`

### Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/lecture-ai.git](https://github.com/yourusername/lecture-ai.git)
   cd lecture-ai
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
3. **Run the application:**
   ```bash
   streamlit run app.py
