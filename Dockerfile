FROM python:3.10-slim

WORKDIR /app

# 1. Install C++ Build Tools (Required for pybind11 and CMake)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Setup Virtual Environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 3. Install basic Python dependencies first (helps with Docker caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# 4. Copy the ENTIRE fast_tokenizer directory into the container
# This ensures setup.py, CMakeLists.txt, and the cpp files are all present
COPY src/fast_tokenizer/ ./src/fast_tokenizer/

# 5. Compile and install your C++ extension locally
# Navigate into the specific folder we just copied and install it
RUN cd src/fast_tokenizer && pip install .

# 6. Pre-download the Hugging Face model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# 7. Copy the rest of your agent's source code
COPY src/rag_code_assistant/agent.py .

# 8. Expose Hugging Face Port
EXPOSE 7860

# 9. Start FastAPI via uvicorn
CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "7860"]