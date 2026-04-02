from dotenv import load_dotenv
import streamlit as st
import os
import base64
import re

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from google.cloud import vision

st.set_page_config(page_title="Ai-Tutor Chatbot", page_icon="🤖", layout="centered")
load_dotenv()

MODEL = os.getenv("LLM_MODEL")
CHROMA_DIR = os.getenv("CHROMA_DB")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:\Seing Ratana Notebook\Chatbot Project\RAG_gpt-4o\Key.json"

st.sidebar.info(f"Current Model: {MODEL}")

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

KHMER_REGEX = re.compile(r"[\u1780-\u17FF]")
MAX_OCR_CHARS = 6000


class E5Embeddings(HuggingFaceEmbeddings):
    def embed_query(self, text: str):
        return super().embed_query(f"query: {text}")

    def embed_documents(self, texts):
        return super().embed_documents([f"passage: {t}" for t in texts])


@st.cache_resource
def get_chroma_instance():
    embedding_function = E5Embeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_function,
        collection_name="khmer_corpus",
        collection_metadata={"hnsw:space": "cosine"},
    )


db = get_chroma_instance()


def _normalize_math_latex(s: str) -> str:
    s = s.replace("```latex", "").replace("```tex", "").replace("```", "")
    s = re.sub(r"\\\((.+?)\\\)", r"$\1$", s, flags=re.S)
    s = re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", s, flags=re.S)

    def _block(m):
        return f"\n\n$$\n{m.group(1).strip()}\n$$\n\n"

    s = re.sub(r"\$\$(.+?)\$\$", _block, s, flags=re.S)
    s = re.sub(r"^[ \t]+(\$\$)\s*$", r"$$", s, flags=re.M)
    return s.strip().replace("\r\n", "\n")


@tool
def query_documents(question: str) -> str:
    """Uses RAG to search Chroma for the most relevant passages to answer the question.

    Args:
        question: The user question to search against the vector DB.
    Returns:
        A newline-joined string of relevant passages, or a message if none found.
    """
    try:
        similar_docs = db.similarity_search(question, k=3)
        if not similar_docs:
            return "No relevant documents found in the knowledge base for this question."
        parts = []
        for doc in similar_docs:
            content = (doc.page_content or "").strip()
            if len(content) > 20:
                parts.append(content)
        if not parts:
            return "Found documents but they appear to be empty or corrupted."
        return "\n\n".join(parts)
    except Exception as e:
        return f"Error accessing knowledge base: {str(e)}"

def prompt_ai(messages, nested_calls=0):
    if nested_calls > 3:
        raise Exception("AI is tool calling too much!")
    if nested_calls == 0 and len(messages) > 4:
        messages = [messages[0]] + messages[-3:]
    
    chatbot = ChatOpenAI(model=MODEL, temperature=0.1, service_tier='default')
    chatbot_with_tools = chatbot.bind_tools([query_documents])

    stream = chatbot_with_tools.stream(messages)
    gathered = None

    for chunk in stream:
        text = getattr(chunk, "content", None)
        if text:
            yield text
        gathered = chunk if gathered is None else (gathered + chunk)

    if gathered and getattr(gathered, "tool_calls", None):
        temp_messages = messages + [gathered]
        for tool_call in gathered.tool_calls:
            if tool_call.get("name") == "query_documents":
                tool_output = query_documents.invoke(tool_call["args"])
                temp_messages.append(
                    ToolMessage(
                        content=tool_output,
                        tool_call_id=tool_call["id"],
                        name="query_documents",
                    )
                )
        yield from prompt_ai(temp_messages, nested_calls + 1)


SYSTEM_MESSAGE = """
You are an AI assistant specializing in Khmer literature and general knowledge. You are fluent in Khmer and English.

Key Rules:
- For all Khmer literature or cultural questions or Khmer name → always use query_documents(question) first.
- Give short, direct with complete sentences. Answer with bullet points if multiple items.
- Khmer question → answer in Khmer. English question → answer in English (use Khmer names correctly).
- If no relevant information found related → reply “អត់មានព័ត៌មានអំពីរឿងនេះទេ” or “No information available.”

Examples:
- “ទុំជានរណា?” → “ទុំ ជាកូនរបស់ស្ត្រីមេម៉ាយ...”
- “Who is Tum?” → “Tum is the son of a widow...”

Math format rule:
- When including equations, write them in LaTeX.
- Inline math -> $ ... $
- Display math -> $$ ... $$
- Do not wrap LaTeX in code blocks.
"""


def make_mm_content(text: str, data_url: str | None):
    parts = [{"type": "text", "text": text}]
    if data_url:
        parts.append({"type": "image_url", "image_url": {"url": data_url, "detail": "high"}})
    return parts


def extract_khmer_text_only(img_bytes: bytes) -> str | None:
    try:
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=img_bytes)
        img_ctx = vision.ImageContext(language_hints=["km"])
        response = client.document_text_detection(image=image, image_context=img_ctx)

        text = ""
        if getattr(response, "full_text_annotation", None) and response.full_text_annotation.text:
            text = response.full_text_annotation.text.strip()
        elif getattr(response, "text_annotations", None):
            if response.text_annotations and response.text_annotations[0].description:
                text = response.text_annotations[0].description.strip()

        if not text or not KHMER_REGEX.search(text):
            return None
        if len(text) > MAX_OCR_CHARS:
            text = text[:MAX_OCR_CHARS] + "\n...[truncated]"
        return text
    except Exception:
        return None


def main():
    st.title("🤖 Ai-Tutor Chatbot")
    st.caption("AI Assistant Ready to Help")

    if "messages" not in st.session_state:
        st.session_state.messages = [SystemMessage(content=SYSTEM_MESSAGE)]

    with st.sidebar:
        st.header("📎 Attach Image")
        f = st.file_uploader(
            "Upload an image",
            type=["png", "jpg", "jpeg"],
            key=f"uploader_{st.session_state.uploader_key}",
        )

        if f is not None:
            img_bytes = f.getvalue()
            st.image(img_bytes, caption=f.name, width='stretch')

            khmer_text = extract_khmer_text_only(img_bytes)
            if khmer_text:
                st.session_state["last_extracted_text"] = khmer_text

            b64 = base64.b64encode(img_bytes).decode("utf-8")
            mime = f.type or "image/png"
            st.session_state["last_uploaded_image_data_url"] = f"data:{mime};base64,{b64}"
            st.session_state.uploader_key += 1

    for m in st.session_state.messages:
        if isinstance(m, HumanMessage):
            with st.chat_message("user"):
                ui_override = getattr(m, "additional_kwargs", {}).get("ui_content", None)
                content_to_show = ui_override if ui_override is not None else m.content

                if isinstance(content_to_show, list):
                    txt = next((p.get("text", "") for p in content_to_show
                                if isinstance(p, dict) and p.get("type") == "text"), "")
                    img = next((((p.get("image_url") or {}).get("url"))
                            for p in content_to_show
                            if isinstance(p, dict) and p.get("type") == "image_url"), None)
                    if txt:
                        st.markdown(txt)
                    if img:
                        st.image(img, width='stretch')
                else:
                    st.markdown(content_to_show)
        elif isinstance(m, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(_normalize_math_latex(m.content))


    prompt = st.chat_input("What would you like to do today?")
    if prompt:
        ocr_text = st.session_state.pop("last_extracted_text", None)
        image_data_url = st.session_state.pop("last_uploaded_image_data_url", None)

        ui_text = prompt
        ui_content = make_mm_content(ui_text, image_data_url) if image_data_url else ui_text

        model_text = prompt
        if ocr_text:
            model_text = f"{prompt}\n\n[Extracted Khmer Text From The Image:\n{ocr_text}]"
        model_content = make_mm_content(model_text, image_data_url) if image_data_url else model_text

        with st.chat_message("user"):
            if isinstance(ui_content, list):
                st.markdown(ui_content[0].get("text", ""))
                if len(ui_content) > 1:
                    st.image(ui_content[1]["image_url"]["url"], width='stretch')
            else:
                st.markdown(ui_content)

        st.session_state.messages.append(
            HumanMessage(
                content=model_content,
                additional_kwargs={"ui_content": ui_content}
            )
        )

        with st.chat_message("assistant"):
            try:
                response = "".join(prompt_ai(st.session_state.messages))
                response = _normalize_math_latex(response)
                st.markdown(response)
            except Exception as e:
                response = f"An error occurred: {e}"
                st.write(response)

        st.session_state.messages.append(AIMessage(content=str(response)))



if __name__ == "__main__":
    main()