import logging
import json
from typing import Dict, Optional
from langchain_core.tools import Tool
from deep_translator import GoogleTranslator

logger = logging.getLogger(__name__)

class TranslationTool:
    """
    A LangChain Tool for translating text between languages using deep_translator's GoogleTranslator.
    """
    def __init__(self, translation_config: Dict):
        self.translation_config = translation_config
        self.google_translator = None
        
        google_trans_conf = self.translation_config.get('google_translate', {})
        if google_trans_conf.get('type') == 'library' and google_trans_conf.get('library') == 'googletrans':
            try:
                self.google_translator = GoogleTranslator()
                logger.info("Google Translator initialized for TranslationTool.")
            except Exception as e:
                logger.warning(f"Failed to initialize Google Translator: {e}.")
        
        logger.info("TranslationTool initialized.")

    def _run(self, json_input: str) -> str:
        try:
            params = json.loads(json_input)
            text_to_translate = params.get('text')
            target_language = params.get('target_language', 'en')

            if not text_to_translate:
                return "Error: 'text' field is missing in input for translation."
            if not target_language:
                return "Error: 'target_language' field is missing in input for translation."

            if self.google_translator:
                try:
                    translated_text = GoogleTranslator(source='auto', target=target_language).translate(text_to_translate)
                    if translated_text:
                        return translated_text
                except Exception as e:
                    logger.warning(f"Google Translator failed: {e}")

            return f"Translation failed for '{text_to_translate}' to '{target_language}'. No active translation service could complete the request."

        except json.JSONDecodeError:
            return "Error: Input for translation_tool must be a valid JSON string with 'text' and 'target_language' keys (e.g., '{\"text\": \"Hello\", \"target_language\": \"km\"}}')."
        except Exception as e:
            logger.error(f"An unexpected error occurred during translation: {e}", exc_info=True)
            return f"An unexpected error occurred during translation: {e}"

    def as_tool(self) -> Tool:
        return Tool(
            name="translation_tool",
            description="Useful for translating text between languages. "
                        "Input should be a JSON string with 'text' and 'target_language' keys "
                        "(e.g., '{\"text\": \"Hello\", \"target_language\": \"km\"}}'). "
                        "Supports common language codes like 'en' (English) and 'km' (Khmer). "
                        "Returns the translated text.",
            func=self._run
        )