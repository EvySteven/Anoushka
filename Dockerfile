FROM python:3.10-slim

WORKDIR /app

# Installer les packages Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier les fichiers du projet (main.py, static/, etc.)
COPY . .

# Exposer le port du serveur
EXPOSE 7860

# Démarrer FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]