from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr
import google.generativeai as genai
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- Constantes ---
KNOWLEDGE_BASE_FILE = "dados.txt"
GEMINI_MODEL_NAME = "gemini-1.0-pro" # Modelo estável recomendado
PROMPT_TEMPLATE = """
Use o conteúdo abaixo como base para responder a pergunta de forma direta, sem inventar nada que não esteja no texto.

=== BASE DE CONHECIMENTO ===
{base_conhecimento}

=== PERGUNTA ===
{pergunta_usuario}

Responda com base apenas no conteúdo da base acima.
"""

# --- Funções do Chatbot (Lógica integrada) ---
def carregar_conhecimento(caminho: str = KNOWLEDGE_BASE_FILE) -> Optional[str]:
    """Carrega a base de conhecimento de um arquivo de texto."""
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logging.error(f"❌ ERRO: Arquivo da base de conhecimento não encontrado em '{caminho}'.")
        return None

def iniciar_gemini() -> Optional[genai.ChatSession]:
    """Configura e inicia uma sessão de chat com a API do Google Gemini."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logging.error("❌ ERRO: API KEY da Gemini não encontrada. Verifique suas variáveis de ambiente.")
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME)
        return model.start_chat()
    except Exception as e:
        logging.error(f"❌ ERRO: Falha ao iniciar o modelo Gemini: {e}")
        return None

def responder_com_gemini(chat: genai.ChatSession, base_conhecimento: str, pergunta_usuario: str) -> str:
    """Envia uma pergunta para o Gemini e retorna a resposta."""
    prompt = PROMPT_TEMPLATE.format(base_conhecimento=base_conhecimento, pergunta_usuario=pergunta_usuario)
    resposta = chat.send_message(prompt)
    return resposta.text.strip()

# --- Configuração do App FastAPI ---
app = FastAPI(
    title="Bytezinho Chatbot API",
    description="API para o chatbot do projeto Jovem Programador.",
    version="1.0.0"
)

# --- Configuração do CORS ---
origins = [
    "https://seu-dominio-na-vercel.app",  # IMPORTANTE: Substitua pela URL do seu frontend
    "http://localhost:3000",
    "http://127.0.0.1:5500"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Inicialização do Chatbot ---
base_conhecimento = None
chat_session = None

@app.on_event("startup")
def startup_event():
    """Executa na inicialização do servidor."""
    global base_conhecimento, chat_session
    logging.info("Iniciando o chatbot...")
    base_conhecimento = carregar_conhecimento()
    chat_session = iniciar_gemini()
    if not chat_session or not base_conhecimento:
        logging.warning("[AVISO] O chatbot está rodando em modo degradado. Verifique a API Key e o arquivo de dados.")
    else:
        logging.info("✅ Chatbot Gemini inicializado com sucesso!")

# --- Modelos de Dados (Pydantic) ---
class ChatRequest(BaseModel):
    message: str

class Lead(BaseModel):
    name: str
    email: EmailStr

# --- Endpoints da API ---
@app.get("/", tags=["Status"])
async def read_root():
    """Endpoint raiz para verificar se a API está no ar."""
    return {"status": "API do Chatbot está funcionando!"}

@app.post("/leads", tags=["Leads"])
async def capturar_lead(lead: Lead):
    """Recebe e armazena um novo lead."""
    logging.info(f"✅ Novo lead recebido: Nome='{lead.name}', Email='{lead.email}'")
    try:
        with open("leads.txt", "a", encoding="utf-8") as f:
            f.write(f"Nome: {lead.name}, Email: {lead.email}\n")
        return {"status": "success", "message": "Lead recebido com sucesso!"}
    except Exception as e:
        logging.error(f"❌ ERRO ao salvar lead: {e}")
        raise HTTPException(status_code=500, detail="Falha ao salvar o lead.")

@app.post("/chat", tags=["Chatbot"])
async def chat(chat_request: ChatRequest):
    """Recebe a mensagem do usuário e retorna a resposta do bot."""
    if not chat_session or not base_conhecimento:
        raise HTTPException(status_code=503, detail="Desculpe, o chatbot não está disponível no momento.")

    user_message = chat_request.message
    if not user_message:
        raise HTTPException(status_code=400, detail="A mensagem não pode ser vazia.")

    try:
        bot_response = responder_com_gemini(chat_session, base_conhecimento, user_message)
        return {'response': bot_response}
    except Exception as e:
        logging.error(f"Erro ao gerar resposta do Gemini: {e}")
        raise HTTPException(status_code=500, detail="Ocorreu um erro ao processar sua mensagem.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
