import sqlite3
import json
from datetime import datetime
import logging
import re
from typing import List, Dict

logger = logging.getLogger(__name__)

class ChatMemory:
    """
    Manages chat history persistence with privacy protection.
    """
    def __init__(self, db_path: str = './data/memory.db'):
        self.db_path = db_path
        self._create_table()
        logger.info(f"ChatMemory initialized with database: {db_path}")

    def _create_table(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()
        logger.debug("Chat history table ensured to exist.")

    def _sanitize_content(self, content: str) -> str:
        """Remove potentially sensitive information from messages"""
        # Redact API keys, tokens, and credentials
        sanitized = re.sub(r'\b(?:password|secret|key|token|credential)\b.*', '[REDACTED]', content, flags=re.IGNORECASE)
        
        # Remove command injection attempts
        sanitized = re.sub(r'[;&|`]', '', sanitized)
        
        return sanitized

    def add_message(self, session_id: str, role: str, content: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        
        # Sanitize content before storage
        sanitized_content = self._sanitize_content(content)
        
        try:
            cursor.execute('''
                INSERT INTO chat_history (session_id, timestamp, role, content)
                VALUES (?, ?, ?, ?)
            ''', (session_id, timestamp, role, sanitized_content))
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error adding message to DB: {e}", exc_info=True)
        finally:
            conn.close()

    def get_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                SELECT role, content FROM chat_history
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (session_id, limit))
            
            # Return in chronological order (oldest first)
            history = cursor.fetchall()
            formatted_history = []
            for row in reversed(history):
                formatted_history.append({"role": row[0], "content": row[1]})
                
            logger.debug(f"Retrieved {len(formatted_history)} messages for session")
            return formatted_history
        except sqlite3.Error as e:
            logger.error(f"Error retrieving history: {e}", exc_info=True)
            return []
        finally:
            conn.close()

    def clear_history(self, session_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute('DELETE FROM chat_history WHERE session_id = ?', (session_id,))
            conn.commit()
            logger.info(f"Cleared chat history for session {session_id}.")
        except sqlite3.Error as e:
            logger.error(f"Error clearing history: {e}", exc_info=True)
        finally:
            conn.close()