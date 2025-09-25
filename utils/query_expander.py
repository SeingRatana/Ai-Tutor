import logging
import re
from typing import Dict, List, Optional
from deep_translator import GoogleTranslator

logger = logging.getLogger(__name__)

class QueryExpander:
    """
    Expands user queries with synonyms and optional translations to improve
    semantic search recall from the knowledge base.
    """
    def __init__(self, itc_synonyms: Dict, ams_synonyms: Dict, translation_config: Dict):
        self.itc_synonyms = itc_synonyms
        self.ams_synonyms = ams_synonyms
        self.translation_config = translation_config
        self.translator = None

        google_trans_conf = self.translation_config.get('google_translate', {})
        if google_trans_conf.get('type') == 'library' and google_trans_conf.get('library') == 'googletrans':
            try:
                self.translator = GoogleTranslator()
                logger.info("Google Translator initialized for QueryExpander.")
            except Exception as e:
                logger.warning(f"Failed to initialize Google Translator: {e}")
        
        logger.info("QueryExpander initialized.")

    def _translate_if_needed(self, text: str, target_lang: str = 'en') -> Optional[str]:
        if self.translator:
            try:
                translated = self.translator.translate(text, target=target_lang)
                if translated and translated.lower().strip() != text.lower().strip():
                    return translated
            except Exception as e:
                logger.error(f"Translation error: {e}", exc_info=True)
        return None

    def expand_query(self, query: str) -> List[str]:
        expanded_queries = [query.lower().strip()]

        translated_query_en = self._translate_if_needed(query, target_lang='en')
        if translated_query_en and translated_query_en not in expanded_queries:
            expanded_queries.append(translated_query_en)
        
        translated_query_km = self._translate_if_needed(query, target_lang='km')
        if translated_query_km and translated_query_km not in expanded_queries:
            expanded_queries.append(translated_query_km)

        current_expanded_set = set(expanded_queries)
        combined_synonyms = {**self.itc_synonyms, **self.ams_synonyms}
        queries_to_process = list(current_expanded_set)

        for q in queries_to_process:
            q_lower = q.lower()
            for entity_key, data in combined_synonyms.items():
                canonical = data.get('canonical', entity_key)
                variations = data.get('variations', [])
                all_forms = [canonical.lower()] + [v.lower() for v in variations]
                
                for form_text in all_forms:
                    pattern = r'\b' + re.escape(form_text) + r'\b'
                    if re.search(pattern, q_lower):
                        for substitute_form_text in all_forms:
                            if substitute_form_text != form_text:
                                new_query = re.sub(pattern, substitute_form_text, q_lower).strip()
                                if new_query and new_query not in current_expanded_set:
                                    current_expanded_set.add(new_query)
                                    queries_to_process.append(new_query)

        final_queries = list(current_expanded_set)
        logger.debug(f"Expanded query '{query}' to: {final_queries}")
        return final_queries