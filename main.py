import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests
import chromadb
import json

# 1. Charger les variables d'environnement (.env en local ou variables Render en Cloud)
load_dotenv()

# 2. Récupérer la clé API Groq et le mot de passe Admin
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("OPENROUTER_API_KEY")
ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY", "ne2026admin")

if not GROQ_API_KEY:
    print("⚠️ ATTENTION : La variable GROQ_API_KEY n'est pas trouvée dans le fichier .env ou Render !")

app = FastAPI(title="Backend Né IA - Groq Cloud")

# Montage du dossier statique pour le CSS / JS / HTML
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# 3. Base de données RAG ChromaDB
chroma_client = chromadb.PersistentClient(path="./knowledge_db")
collection = chroma_client.get_or_create_collection(name="sante_femmes")

PENDING_FILE = "propositions_en_attente.json"
if not os.path.exists(PENDING_FILE):
    with open(PENDING_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

# Modèles de données Pydantic
class QuestionRequest(BaseModel):
    message: str

class PropositionRequest(BaseModel):
    titre: str
    contenu: str
    auteur: str = "Anonyme"

class ValidationRequest(BaseModel):
    index: int
    secret_key: str

class AjoutDirectRequest(BaseModel):
    titre: str
    contenu: str
    secret_key: str

# ------------------------------------------------------------------
# ROUTES WEB (HTML)
# ------------------------------------------------------------------
@app.get("/")
def page_publique():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"status": "API Né IA fonctionnelle", "docs": "/docs"}

@app.get("/admin")
def page_admin():
    if os.path.exists("static/admin.html"):
        return FileResponse("static/admin.html")
    return {"status": "Espace Admin - fichier static/admin.html introuvable"}

# ------------------------------------------------------------------
# ENDPOINT CHAT (GROQ API - LLAMA 3.3 70B ULTRA RAPIDE)
# ------------------------------------------------------------------
@app.post("/api/chat")
def chat(req: QuestionRequest):
    # 1. Recherche RAG dans la mémoire ChromaDB
    results = collection.query(query_texts=[req.message], n_results=2)
    documents = results.get("documents", [[]])[0]
    contexte_trouve = "\n---\n".join(documents) if documents else "Aucun document spécifique trouvé."

    prompt_systeme = (
        "Tu es Anoushka, l'assistante santé féminine bienveillante intégrée à l'application Né. "
        "Voici les connaissances médicales vérifiées de notre base :\n"
        f"{contexte_trouve}\n\n"
        "Réponds à l'utilisatrice avec empathie, clarté et concision."
    )

    # 2. Appel à l'API Groq
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",  # Modèle gratuit ultra-rapide sur Groq
        "messages": [
            {"role": "system", "content": prompt_systeme},
            {"role": "user", "content": req.message}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        res_data = response.json()
        
        print("=== REPONSE GROQ API ===", res_data)

        if 'choices' in res_data and len(res_data['choices']) > 0:
            reponse_texte = res_data['choices'][0]['message']['content']
        elif 'error' in res_data:
            reponse_texte = f"Erreur Groq API: {res_data['error'].get('message', 'Clé API invalide ou quota dépassé.')}"
        else:
            reponse_texte = "Désolé, l'IA est momentanément indisponible."

        return {"reponse": reponse_texte, "source": contexte_trouve}
    except Exception as e:
        print("Erreur exception Groq:", str(e))
        raise HTTPException(status_code=500, detail=f"Erreur API Groq: {str(e)}")

# Alias de compatibilité pour la route /chat
@app.post("/chat")
def chat_alias(req: QuestionRequest):
    return chat(req)

# ------------------------------------------------------------------
# ENDPOINT PROPOSITION COMMUNAUTAIRE
# ------------------------------------------------------------------
@app.post("/api/proposer")
def proposer(prop: PropositionRequest):
    with open(PENDING_FILE, "r", encoding="utf-8") as f:
        props = json.load(f)
    props.append({"titre": prop.titre, "contenu": prop.contenu, "auteur": prop.auteur})
    with open(PENDING_FILE, "w", encoding="utf-8") as f:
        json.dump(props, f, ensure_ascii=False, indent=2)
    return {"message": "Proposition enregistrée."}

# ------------------------------------------------------------------
# ENDPOINTS ADMIN (MODÉRATION & RAG)
# ------------------------------------------------------------------
@app.get("/api/admin/attente")
def voir_attente(key: str = Query(...)):
    if key != ADMIN_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Accès non autorisé")
    with open(PENDING_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

@app.post("/api/admin/valider")
def valider(req: ValidationRequest):
    if req.secret_key != ADMIN_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Accès non autorisé")
    
    with open(PENDING_FILE, "r", encoding="utf-8") as f:
        props = json.load(f)
    if req.index < 0 or req.index >= len(props):
        raise HTTPException(status_code=400, detail="Index invalide")
    
    item = props.pop(req.index)
    doc_id = f"doc_{len(collection.get()['ids']) + 1}"
    collection.add(
        documents=[f"Titre: {item['titre']}\nContenu: {item['contenu']}"],
        metadatas=[{"titre": item['titre'], "auteur": item['auteur']}],
        ids=[doc_id]
    )
    with open(PENDING_FILE, "w", encoding="utf-8") as f:
        json.dump(props, f, ensure_ascii=False, indent=2)
    return {"message": f"Connaissance '{item['titre']}' validée !"}

@app.delete("/api/admin/rejeter")
def rejeter(index: int, key: str = Query(...)):
    if key != ADMIN_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Accès non autorisé")
    with open(PENDING_FILE, "r", encoding="utf-8") as f:
        props = json.load(f)
    if index >= 0 and index < len(props):
        props.pop(index)
    with open(PENDING_FILE, "w", encoding="utf-8") as f:
        json.dump(props, f, ensure_ascii=False, indent=2)
    return {"message": "Proposition rejetée."}

@app.get("/api/admin/connaissances")
def voir_connaissances(key: str = Query(...)):
    if key != ADMIN_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Accès non autorisé")
    data = collection.get()
    resultats = []
    if data and data['documents']:
        for i in range(len(data['documents'])):
            metas = data['metadatas'][i] if data['metadatas'] else {}
            resultats.append({
                "id": data['ids'][i],
                "titre": metas.get('titre', 'Sans titre'),
                "contenu": data['documents'][i]
            })
    return resultats

@app.post("/api/admin/ajouter-direct")
def ajouter_direct(req: AjoutDirectRequest):
    if req.secret_key != ADMIN_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Accès non autorisé")
    doc_id = f"doc_{len(collection.get()['ids']) + 1}"
    collection.add(
        documents=[f"Titre: {req.titre}\nContenu: {req.contenu}"],
        metadatas=[{"titre": req.titre, "auteur": "Admin"}],
        ids=[doc_id]
    )
    return {"message": "Ajouté à l'IA."}

@app.delete("/api/admin/supprimer-connaissance")
def supprimer_connaissance(doc_id: str, key: str = Query(...)):
    if key != ADMIN_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Accès non autorisé")
    collection.delete(ids=[doc_id])
    return {"message": "Effacé de la mémoire de l'IA."}