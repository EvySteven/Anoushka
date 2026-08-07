import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests
import chromadb
import json

# 1. Charger les variables d'environnement
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("OPENROUTER_API_KEY")
ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY", "ne2026admin")

app = FastAPI(title="Backend Né IA - Mémoire & Conversation")

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# 2. Base RAG ChromaDB
chroma_client = chromadb.PersistentClient(path="./knowledge_db")
collection = chroma_client.get_or_create_collection(name="sante_femmes")

PENDING_FILE = "propositions_en_attente.json"
if not os.path.exists(PENDING_FILE):
    with open(PENDING_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

# 🧠 MÉMOIRE DES CONVERSATIONS EN MÉMOIRE SERVEUR
# Format : { "session_id": [ {"role": "user/assistant", "content": "..."}, ... ] }
SESSIONS = {}

# Modèles Pydantic
class QuestionRequest(BaseModel):
    message: str
    session_id: str = "default_session" # ID unique de l'utilisatrice
    prenom: str = None                 # Prénom optionnel

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
# ENDPOINT CHAT AVEC MÉMOIRE & PERSONNALITÉ
# ------------------------------------------------------------------
@app.post("/api/chat")
def chat(req: QuestionRequest):
    session_id = req.session_id

    # 1. Initialiser la mémoire de la session si elle n'existe pas encore
    if session_id not in SESSIONS:
        SESSIONS[session_id] = []

    # 2. Recherche RAG dans ChromaDB pour le contexte médical
    contexte_trouve = "Aucun document spécifique trouvé."
    try:
        if collection.count() > 0:
            results = collection.query(query_texts=[req.message], n_results=2)
            documents = results.get("documents", [[]])[0]
            if documents:
                contexte_trouve = "\n---\n".join(documents)
    except Exception as e:
        print("Avertissement ChromaDB:", e)

    # 3. Prompt de personnalité chaleureuse et amicale
    prenom_str = f"L'utilisatrice s'appelle {req.prenom}. " if req.prenom else ""

    prompt_systeme = f"""Tu es Anoushka, une compagne et assistante amicale, chaleureuse et bienveillante intégrée à l'application Né. {prenom_str}
Tu réponds principalement sur la santé sexuelle et reproductive, mais tu peux aussi discuter de TOUT avec plaisir (vie quotidienne, météo, conseils, culture, voyages).

GESTION DES LANGUES :
- Par défaut, réponds en français fluide, naturel et impeccablement correct (sans aucune faute de grammaire).
- Tu es polyglotte : si l'utilisatrice s'adresse à toi ou te demande de répondre dans une langue internationale (Anglais, Espagnol, Chinois/Mandarin, Arabe, Allemand, Portugais, etc.), adapte-toi immédiatement et réponds couramment dans la langue demandée.

ADAPTATION AU JARGON ET STYLE DE L'UTILISATRICE :
- Adapte-toi naturellement au jargon, au niveau de langage et aux expressions de l'utilisatrice (effet miroir).
- Si elle utilise du jargon jeune, des expressions locales décontractées ou familiales, réponds-lui avec la même proximité et complicité, sans jamais perdre en clarté ni en correction de grammaire.
- Si elle utilise un langage plus formel ou technique, adapte-toi avec le même niveau de précision.

TON ET STYLE DE CONVERSATION :
- Sois chaleureuse, amicale, pétillante et parfois drôle pour les discussions quotidiennes. Tu peux utiliser des émojis 😊 pour rendre l'échange chaleureux !
- Reste douce, empathique et sérieuse si l'utilisatrice te parle d'un problème de santé triste ou douloureux.
- Tutoie l'utilisatrice de façon amicale (ou vouvoie si elle le demande).

RÈGLES D'OR DE TON COMPORTEMENT :
1. NE TE PRÉSENTE PAS à chaque message. Ne dis PAS 'Bonjour, je suis Anoushka...' à chaque réponse ! Réponds directement et naturellement.
2. Tu donneras ton nom uniquement lors de la toute première prise de contact si nécessaire.
3. Si l'utilisatrice te donne son prénom ou te demande si tu t'en souviens, retiens-le et utilise-le chaleureusement.
4. Quand une question porte sur la santé, appuie-toi sur ces données vérifiées si utile :
{contexte_trouve}
5. Reste concise, empathique et naturelle (2 à 4 phrases sauf si une longue explication est demandée).
6. Ne donne JAMAIS de conseils médicaux précis ou de prescriptions, et n'invente jamais de faits médicaux. Encourage toujours à consulter un professionnel de santé.
7. Si tu ne sais pas une information, dis-le honnêtement et propose de reformuler ou de chercher.
"""
    # 4. Reconstruire l'historique complet pour Groq
    messages_payload = [{"role": "system", "content": prompt_systeme}]
    
    # Ajouter les anciens messages de la conversation (max 10 derniers échanges)
    historique_recent = SESSIONS[session_id][-10:]
    messages_payload.extend(historique_recent)
    
    # Ajouter le nouveau message de l'utilisatrice
    messages_payload.append({"role": "user", "content": req.message})

    # 5. Appel Groq API (Ultra-rapide)
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": messages_payload,
        "max_tokens": 400,
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        res_data = response.json()

        if 'choices' in res_data and len(res_data['choices']) > 0:
            reponse_texte = res_data['choices'][0]['message']['content']
            
            # 🧠 SAUVEGARDER DANS LA MÉMOIRE DE LA SESSION
            SESSIONS[session_id].append({"role": "user", "content": req.message})
            SESSIONS[session_id].append({"role": "assistant", "content": reponse_texte})
        elif 'error' in res_data:
            reponse_texte = f"Erreur Groq: {res_data['error'].get('message', 'Clé API invalide.')}"
        else:
            reponse_texte = "Désolé, l'IA est momentanément indisponible."

        return {"reponse": reponse_texte, "source": contexte_trouve}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur API Groq: {str(e)}")

# Alias de route
@app.post("/chat")
def chat_alias(req: QuestionRequest):
    return chat(req)

# ------------------------------------------------------------------
# AUTRES ENDPOINTS API (MODÉRATION & ADMIN)
# ------------------------------------------------------------------
@app.post("/api/proposer")
def proposer(prop: PropositionRequest):
    with open(PENDING_FILE, "r", encoding="utf-8") as f:
        props = json.load(f)
    props.append({"titre": prop.titre, "contenu": prop.contenu, "auteur": prop.auteur})
    with open(PENDING_FILE, "w", encoding="utf-8") as f:
        json.dump(props, f, ensure_ascii=False, indent=2)
    return {"message": "Proposition enregistrée."}

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