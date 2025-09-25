import sys
import os
from pathlib import Path
import yaml
import logging
from dotenv import load_dotenv
from urllib.parse import urlparse
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from vector_store.chroma_db import ChromaDB
from vector_store.embedder import Embedder
from utils.document_processor import DocumentProcessor

def run_initial_ingestion():
    logger.info("Starting initial data ingestion process...")

    try:
        with open('config/api_keys.yaml', 'r', encoding='utf-8') as f:
            api_config = yaml.safe_load(f)
        logger.info("config/api_keys.yaml loaded.")
    except (FileNotFoundError, yaml.YAMLError) as e:
        logger.error(f"Failed to load config: {e}")
        return

    embedder_config = api_config['embedding_services']['sentence_transformers']
    embedder = Embedder(model_name=embedder_config['model'])
    embedding_function = embedder.get_embedding_function()

    chroma_dir = api_config['data_paths']['chroma_db_dir']
    os.makedirs(chroma_dir, exist_ok=True)
    vector_db = ChromaDB(persist_directory=chroma_dir, embedding_function=embedding_function)
    logger.info("ChromaDB initialized for data ingestion.")

    doc_processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    logger.info("DocumentProcessor initialized.")

    itc_urls = api_config['domain_filters'].get('itc', [])
    ams_urls = api_config['domain_filters'].get('ams', [])
    
    all_target_urls_for_scraping = []
    for url_pattern in itc_urls + ams_urls:
        if url_pattern.endswith('**'):
            base_url = url_pattern[:-2]
            parsed_base = urlparse(base_url)
            if not parsed_base.path and not base_url.endswith('/'):
                base_url += '/'
            all_target_urls_for_scraping.append(base_url)
        else:
            all_target_urls_for_scraping.append(url_pattern)

    all_target_urls_for_scraping = list(set(all_target_urls_for_scraping))
    logger.info(f"Identified {len(all_target_urls_for_scraping)} unique base URLs for ingestion")

    ingested_chunks_count = 0
    for url in all_target_urls_for_scraping:
        logger.info(f"Processing URL: {url}")
        try:
            extracted_text = doc_processor.scrape_and_extract_content(
                url=url,
                user_agent=api_config['web_scraping']['user_agent'],
                max_depth=2
            )
            
            if extracted_text and extracted_text.strip():
                chunks = doc_processor.chunk_text(extracted_text, metadata={"source": url, "ingested_at": datetime.now().isoformat()})
                
                if chunks:
                    documents_to_add_to_db = [{"text": chunk['text'], "metadata": chunk['metadata']} for chunk in chunks]
                    vector_db.add_documents(documents_to_add_to_db)
                    ingested_chunks_count += len(chunks)
                    logger.info(f"Successfully ingested {len(chunks)} chunks from: {url}")
                else:
                    logger.warning(f"No chunks generated from: {url}")
            else:
                logger.warning(f"No content extracted from: {url}")
        except Exception as e:
            logger.error(f"Error ingesting content: {e}", exc_info=True)
    
    logger.info(f"Initial data ingestion complete. Total chunks indexed: {ingested_chunks_count}")
    logger.info(f"Total documents in vector database: {vector_db.count_documents()}")

if __name__ == '__main__':
    run_initial_ingestion()