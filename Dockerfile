FROM python:3.10-slim

# Installer zsh, curl et les dépendances système
RUN apt-get update && apt-get install -y curl procps && rm -rf /var/lib/apt/lists/*

# Installer Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Télécharger le modèle Qwen 2.5 (1.5B ultra-rapide) au moment de la construction
RUN ollama serve & sleep 5 && ollama pull qwen2.5:1.5b

# Port standard de Hugging Face
EXPOSE 7860

# Lancer Ollama + FastAPI au démarrage du serveur
CMD ["sh", "-c", "ollama serve & sleep 5 && uvicorn main:app --host 0.0.0.0 --port 7860"]