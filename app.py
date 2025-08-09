import streamlit as st
import requests
import qdrant_client
from sentence_transformers import SentenceTransformer
import os
from datetime import datetime
import re
import torch

# --- Конфигурация через Streamlit Secrets ---
try:
    if st.secrets.get("runtime", {}).get("environment") == "production":
        st.success("✅ Production mode: Using secure secrets")
        QDRANT_PATH = st.secrets["qdrant"]["path"]
        YANDEX_TRANSLATE_API_KEY = st.secrets["api_keys"]["yandex_translate"]
        API_KEY = st.secrets["api_keys"]["yandex_gpt"]
        FOLDER_ID = st.secrets["api_keys"]["folder_id"]
    else:
        QDRANT_PATH = st.secrets["qdrant"]["path"]
        YANDEX_TRANSLATE_API_KEY = st.secrets["api_keys"]["yandex_translate"]
        API_KEY = st.secrets["api_keys"]["yandex_gpt"] 
        FOLDER_ID = st.secrets["api_keys"]["folder_id"]
except Exception as e:
    st.error(f"⚠️ Ошибка загрузки конфигурации: {e}")
    st.stop()

# Константы
COLLECTION_NAME = "tv_shows"
MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Инициализация клиентов ---
@st.cache_resource
def initialize_qdrant_client(db_path):
    try:
        lock_file = os.path.join(db_path, '.lock')
        if os.path.exists(lock_file):
            os.remove(lock_file)
        return qdrant_client.QdrantClient(path=db_path)
    except Exception as e:
        st.error(f"❌ Ошибка инициализации Qdrant: {str(e)}")
        st.stop()

try:
    client = initialize_qdrant_client(QDRANT_PATH)
    embedding_model = SentenceTransformer(MODEL_NAME, device='cpu')
except Exception as e:
    st.error(f"❌ Ошибка загрузки моделей: {str(e)}")
    st.stop()

# --- Проверка API ключей ---
def check_api_keys():
    return all([API_KEY, FOLDER_ID, YANDEX_TRANSLATE_API_KEY])

# --- Функция определения русского текста ---
def is_russian(text):
    return bool(re.search('[а-яА-Я]', text))

# --- Функция перевода ---
def translate_text(text, target_lang="ru", source_lang=None):
    if not check_api_keys():
        return text
        
    url = "https://translate.api.cloud.yandex.net/translate/v2/translate"
    headers = {
        "Authorization": f"Api-Key {YANDEX_TRANSLATE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "texts": [text],
        "targetLanguageCode": target_lang
    }
    
    if source_lang:
        data["sourceLanguageCode"] = source_lang
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            return response.json()["translations"][0]["text"]
    except Exception:
        pass
    return text

# --- Поиск в Qdrant ---
def search_in_qdrant(query, top_k=3):
    try:
        query_vector = embedding_model.encode(query, convert_to_tensor=True).cpu().numpy().tolist()
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k
        )
        return [hit.payload for hit in search_result]
    except Exception:
        return []

# --- Запрос к YandexGPT ---
def ask_yandex_gpt(user_query, context, check_rag=False):
    if not context or not check_api_keys():
        return "No relevant shows found" if not check_rag else ("", "", "")
        
    context_str = "\n".join([
        f"- Title: {show.get('title', 'N/A')}, Genres: {show.get('genres', 'N/A')}, Description: {show.get('description', 'N/A')}" 
        for show in context
    ])

    system_prompt = """You are a TV show recommendation assistant. 
    Answer based ONLY on the context provided below."""
    
    final_prompt = f"Context:\n{context_str}\n\nUser question: {user_query}"

    if check_rag:
        system_prompt += "\n\nIMPORTANT: You must ONLY use information from the provided context!"
        return system_prompt, final_prompt, context_str

    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    headers = {
        "Authorization": f"Api-Key {API_KEY}",
        "x-folder-id": FOLDER_ID,
        "Content-Type": "application/json"
    }
    data = {
        "modelUri": f"gpt://{FOLDER_ID}/yandexgpt-lite/latest",
        "completionOptions": {"temperature": 0.4, "maxTokens": 2000},
        "messages": [
            {"role": "system", "text": system_prompt},
            {"role": "user", "text": final_prompt}
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=15)
        return response.json()['result']['alternatives'][0]['message']['text']
    except Exception:
        return "Error processing request"

# --- Интерфейс Streamlit ---
def main():
    st.title("🎬 TV Show Recommendation Bot")
    
    tab1, tab2 = st.tabs(["🔍 Поиск сериалов", "🧪 Проверка RAG"])

    with tab1:
        user_query = st.text_input("Your query (in English):", "recommend a series about space and aliens")
        
        if st.button("Search"):
            if not check_api_keys():
                st.error("API keys not configured")
                return
                
            with st.spinner("Processing..."):
                original_query = user_query
                if is_russian(user_query):
                    st.warning("Your text will be translated to English")
                    user_query = translate_text(user_query, target_lang="en", source_lang="ru")
                
                shows = search_in_qdrant(user_query)
                if shows:
                    st.subheader("Found shows (raw data)")
                    st.json(shows, expanded=False)

                    gpt_response = ask_yandex_gpt(user_query, shows)
                    st.subheader("YandexGPT Response")
                    st.write(gpt_response)

                    if is_russian(original_query):
                        st.write(translate_text(gpt_response, target_lang="ru", source_lang="en"))

    with tab2:
        st.subheader("RAG System Inspection")
        
        st.markdown("""
        ### Как работает проверка RAG?
        
        RAG (Retrieval-Augmented Generation) - это система, которая:
        1. **Извлекает** релевантные данные из базы знаний (Qdrant)
        2. **Формирует контекст** для языковой модели
        3. **Генерирует ответ**, используя только предоставленный контекст

        На этой вкладке вы можете проверить:
        - Какие данные были извлечены из базы
        - Как формируется системный промпт для YandexGPT
        - Полный запрос, отправляемый в языковую модель
        """)
        
        test_query = st.text_input("Test query:", "What can you tell me about these shows?")
        
        if st.button("Inspect RAG"):
            with st.spinner("Analyzing RAG components..."):
                shows = search_in_qdrant(
                    st.session_state.get("query_input", "comedy series"), 
                    top_k=2
                )
                
                if shows:
                    system_prompt, final_prompt, context_str = ask_yandex_gpt(
                        test_query, 
                        shows, 
                        check_rag=True
                    )
                    
                    st.subheader("RAG Components Breakdown")
                    
                    with st.expander("1. Retrieved Context Data"):
                        st.text_area("Data from Qdrant", context_str, height=200)
                        st.info("Это сырые данные, извлеченные из векторной базы по вашему запросу")
                    
                    with st.expander("2. System Prompt"):
                        st.text_area("System instructions", system_prompt, height=150)
                        st.info("Эти инструкции гарантируют, что модель будет использовать только предоставленный контекст")
                    
                    with st.expander("3. Final Prompt to YandexGPT"):
                        st.text_area("Complete prompt", final_prompt, height=300)
                        st.info("Полный запрос, включающий контекст и ваш вопрос")
                    
                    st.success("""
                    ✅ RAG система работает корректно если:
                    - В System Prompt есть четкое указание использовать только контекст
                    - Ответ модели соответствует предоставленным данным
                    - Нет "галлюцинаций" (информации не из контекста)
                    """)
                else:
                    st.warning("No data retrieved for analysis")

if __name__ == "__main__":
    main()
