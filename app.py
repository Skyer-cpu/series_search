import streamlit as st
import requests
import qdrant_client
from sentence_transformers import SentenceTransformer
import os
from datetime import datetime

# --- Конфигурация через Streamlit Secrets ---
try:
    # Проверяем, запущено ли приложение в Streamlit Cloud
    if st.secrets.get("runtime", {}).get("environment") == "production":
        st.success("✅ Production mode: Using secure secrets")
        
        # Загрузка конфигурации из secrets.toml
        QDRANT_PATH = st.secrets["qdrant"]["path"]
        YANDEX_TRANSLATE_API_KEY = st.secrets["api_keys"]["yandex_translate"]
        API_KEY = st.secrets["api_keys"]["yandex_gpt"]
        FOLDER_ID = st.secrets["api_keys"]["folder_id"]
    else:
        # Локальная разработка (используем .streamlit/secrets.toml)
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
    st.write("🔹 **Инициализация Qdrant клиента**")
    st.write("1. Проверяем наличие файла блокировки...")
    lock_file = os.path.join(db_path, '.lock')
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
            st.write("   ✅ Файл блокировки удален")
        except OSError as e:
            st.warning(f"   ⚠️ Ошибка удаления файла блокировки: {e}")
    st.write("2. Подключаемся к базе Qdrant...")
    return qdrant_client.QdrantClient(path=db_path)

# Инициализируем клиент
client = initialize_qdrant_client(QDRANT_PATH)
embedding_model = SentenceTransformer(MODEL_NAME)

# --- Проверка API ключей ---
def check_api_keys():
    if not all([API_KEY, FOLDER_ID, YANDEX_TRANSLATE_API_KEY]):
        st.error("❌ Не все API ключи настроены! Проверьте secrets.toml")
        return False
    return True

# --- Функция перевода ---
def translate_text(text, target_lang="ru"):
    if not check_api_keys():
        return text
        
    st.write("🔹 **Запрос к Yandex Translate API**")
    url = "https://translate.api.cloud.yandex.net/translate/v2/translate"
    headers = {
        "Authorization": f"Api-Key {YANDEX_TRANSLATE_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "texts": [text],
        "targetLanguageCode": target_lang
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            st.write("   ✅ Перевод успешно выполнен")
            return response.json()["translations"][0]["text"]
        else:
            st.error(f"   ❌ Ошибка API: {response.status_code}")
    except Exception as e:
        st.error(f"   ❌ Ошибка соединения: {str(e)}")
    
    return text

# --- Поиск в Qdrant ---
def search_in_qdrant(query, top_k=3):
    st.write("🔹 **Векторизация запроса**")
    query_vector = embedding_model.encode(query).tolist()
    
    st.write("🔹 **Поиск в Qdrant**")
    st.write(f"Ищем {top_k} ближайших соседей для запроса: '{query}'")
    
    try:
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k
        )
        st.write(f"   ✅ Найдено {len(search_result)} результатов")
        return [hit.payload for hit in search_result]
    except Exception as e:
        st.error(f"❌ Ошибка поиска в Qdrant: {str(e)}")
        return []

# --- Запрос к YandexGPT ---
def ask_yandex_gpt(user_query, context, check_rag=False):
    if not context:
        return "No relevant shows found."
    
    if not check_api_keys():
        return "API keys not configured"
        
    st.write("🔹 **Формирование контекста для YandexGPT**")
    context_str = "\n".join([
        f"- Title: {show.get('title', 'N/A')}, Genres: {show.get('genres', 'N/A')}, Description: {show.get('description', 'N/A')}" 
        for show in context
    ])

    system_prompt = """You are a TV show recommendation assistant. 
    Answer based ONLY on the context provided below. 
    If you don't know the answer, say 'I don't have enough information'."""
    
    final_prompt = f"Context:\n{context_str}\n\nUser question: {user_query}"

    if check_rag:
        system_prompt += "\n\nIMPORTANT: You must ONLY use information from the provided context!"
        return system_prompt, final_prompt, context_str

    st.write("🔹 **Запрос к YandexGPT API**")
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
        response.raise_for_status()
        st.write("   ✅ Ответ от YandexGPT получен")
        return response.json()['result']['alternatives'][0]['message']['text']
    except Exception as e:
        st.error(f"❌ Ошибка запроса к YandexGPT: {str(e)}")
        return f"Error: {str(e)}"

# --- Интерфейс Streamlit ---
def main():
    st.title("🎬 TV Show Recommendation Bot (Secure)")
    st.warning("⚠️ База данных на английском! Вводите запрос на английском (например: 'comedy about space')")
    
    # Проверка даты ротации ключей
    if "last_key_rotation" not in st.session_state:
        st.session_state.last_key_rotation = datetime(2025, 9, 8)
    
    if (datetime.now() - st.session_state.last_key_rotation).days > 90:
        st.warning(f"🚨 Рекомендуется сменить API ключи! Последняя ротация: {st.session_state.last_key_rotation.strftime('%d.%m.%Y')}")

    tab1, tab2 = st.tabs(["🔍 Поиск сериалов", "🧪 Проверка RAG"])

    with tab1:
        user_query = st.text_input("**Your query (in English):**", "recommend a series about space and aliens", key="query_input")

        if st.button("Search", key="search_btn"):
            if not check_api_keys():
                st.error("Пожалуйста, настройте API ключи в secrets.toml")
                return
                
            with st.spinner("Обработка запроса..."):
                # Шаг 1: Поиск в Qdrant
                st.subheader("🔍 Найденные сериалы (сырые данные)")
                shows = search_in_qdrant(user_query)
                if shows:
                    st.json(shows, expanded=True)

                    # Шаг 2: Запрос к YandexGPT
                    st.subheader("🤖 Ответ YandexGPT")
                    gpt_response_en = ask_yandex_gpt(user_query, shows)
                    st.text_area("Ответ на английском", gpt_response_en, height=200)

                    # Шаг 3: Перевод на русский
                    st.subheader("🇷🇺 Перевод ответа")
                    gpt_response_ru = translate_text(gpt_response_en)
                    st.text_area("Ответ на русском", gpt_response_ru, height=200)
                else:
                    st.warning("Не удалось найти сериалы по вашему запросу")

    with tab2:
        st.subheader("Проверка RAG-системы")
        st.write("Здесь вы можете проверить, использует ли YandexGPT только предоставленный контекст")
        
        test_query = st.text_input("Тестовый запрос:", "What can you tell me about these shows?", key="test_query")
        
        if st.button("Проверить RAG", key="rag_btn"):
            if not check_api_keys():
                st.error("Пожалуйста, настройте API ключи в secrets.toml")
                return
                
            with st.spinner("Анализ RAG-системы..."):
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
                    
                    st.subheader("🔧 Компоненты RAG")
                    with st.expander("System Prompt"):
                        st.text_area("Системный промпт", system_prompt, height=150)
                    
                    with st.expander("Контекст из базы"):
                        st.text_area("Данные из Qdrant", context_str, height=300)
                    
                    with st.expander("Финальный промпт"):
                        st.text_area("Полный промпт для YandexGPT", final_prompt, height=400)
                    
                    st.success("Проверьте, что system prompt явно требует использовать только контекст!")
                else:
                    st.warning("Не удалось загрузить данные для проверки RAG")

if __name__ == "__main__":
    main()