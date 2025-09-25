import os
import asyncio
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify, make_response
import yaml
import logging
import uuid
import re
import json  # Added JSON import
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24).hex())
if not os.getenv("FLASK_SECRET_KEY"):
    logger.warning("FLASK_SECRET_KEY not set in .env. Using a randomly generated one for this session.")

# Load configuration
try:
    with open('config/api_keys.yaml', 'r', encoding='utf-8') as f:
        api_config = yaml.safe_load(f)
    logger.info("config/api_keys.yaml loaded successfully.")
    
    with open('config/synonyms.yaml', 'r', encoding='utf-8') as f:
        synonym_config = yaml.safe_load(f)
    logger.info("config/synonyms.yaml loaded successfully.")
except (FileNotFoundError, yaml.YAMLError) as e:
    logger.error(f"Error loading config files: {e}")
    exit(1)

# Import components after config is loaded
from agents.langchain_agent_orchestrator import LangChainAgentOrchestrator
from utils.memory import ChatMemory

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")

if not GEMINI_API_KEY:
    logger.error("Gemini API key not found in .env.")
    exit(1)

# Initialize with persistent paths
memory = ChatMemory(db_path=api_config['data_paths']['memory_db'])
logger.info(f"Chat Memory initialized: {api_config['data_paths']['memory_db']}")

try:
    agent_orchestrator = LangChainAgentOrchestrator(
        gemini_api_key=GEMINI_API_KEY,
        gemini_model_name=GEMINI_MODEL_NAME,
        api_config=api_config,
        synonym_config=synonym_config
    )
    logger.info("LangGraph Orchestrator initialized successfully.")
except Exception as e:
    logger.critical(f"Failed to initialize LangGraph Orchestrator: {e}", exc_info=True)
    exit(1)

def _is_malicious_query(query: str) -> bool:
    """Detect and block potentially harmful queries"""
    malicious_patterns = [
        r'\b(?:download|export|dump)\b.*\b(?:database|knowledge base|data|vector)\b',
        r'\b(?:system|file|source code|config|\.env|api key|secret)\b',
        r'\b(?:hack|exploit|vulnerability|bypass|security|inject)\b',
        r'\b(?:delete|remove|destroy|erase|reset|drop)\b.*\b(?:data|database|system)\b',
        r'\b(?:password|secret|key|credentials|login)\b',
        r'[;&|`]',  # Command injection characters
        r'(?:\.\./)+',  # Path traversal
        r'\b(?:privacy|private|confidential)\b'
    ]
    
    query_lower = query.lower()
    for pattern in malicious_patterns:
        if re.search(pattern, query_lower):
            return True
    return False

def _log_security_event(event_type: str, details: str):
    """Log security events to a dedicated file"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "type": event_type,
        "details": details
    }
    
    try:
        with open("security.log", "a") as log_file:
            log_file.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to log security event: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
async def search():
    user_query = request.form.get('query')
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # Security check
    if _is_malicious_query(user_query):
        logger.warning(f"Blocked potentially malicious query: {user_query}")
        _log_security_event(
            "BLOCKED_QUERY", 
            f"Query: '{user_query}'"
        )
        return jsonify({
            "answer": "I'm sorry, I cannot assist with that request as it violates security policies. Please ask about ITC/AMS topics.",
            "summary": "Security restriction",
            "details": "Your query appears to be attempting restricted actions.",
            "list_items": [],
            "sources": []
        }), 403

    session_id = request.cookies.get('session_id') or str(uuid.uuid4())
    logger.info(f"Using session: {session_id}")

    chat_history = memory.get_history(session_id)
    logger.info(f"Retrieved {len(chat_history)} messages for session")

    try:
        agent_response_data = await agent_orchestrator.run_agent(user_query, chat_history)
        
        # Store conversation
        memory.add_message(session_id, "user", user_query)
        memory.add_message(session_id, "assistant", json.dumps(agent_response_data))
        
        resp = make_response(jsonify(agent_response_data))
        resp.set_cookie('session_id', session_id, httponly=True, samesite='Lax', max_age=3600*24*7)
        return resp

    except Exception as e:
        logger.error(f"Error during search: {e}", exc_info=True)
        resp = make_response(jsonify({
            "answer": "An internal error occurred. Please try again.",
            "summary": "Processing error",
            "details": f"Technical details: {str(e)}",
            "list_items": [],
            "sources": []
        }))
        resp.set_cookie('session_id', session_id, httponly=True, samesite='Lax', max_age=3600*24*7)
        return resp

@app.route('/history', methods=['GET'])
def get_history():
    session_id = request.cookies.get('session_id')
    if not session_id:
        return jsonify([])
    return jsonify(memory.get_history(session_id))

@app.route('/clear_history', methods=['POST'])
def clear_history():
    session_id = request.cookies.get('session_id')
    if session_id:
        memory.clear_history(session_id)
        return jsonify({"status": "success"})
    return jsonify({"status": "no session to clear"})

if __name__ == '__main__':
    # Ensure data directories exist
    os.makedirs(api_config['data_paths']['chroma_db_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(api_config['data_paths']['memory_db']), exist_ok=True)
    
    logger.info("Starting Flask application")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
