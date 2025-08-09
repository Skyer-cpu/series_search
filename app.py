import streamlit as st
import requests
import qdrant_client
from sentence_transformers import SentenceTransformer
import os
from datetime import datetime
import re
import torch

# --- Настройка стилей ---
def setup_styles():
    st.set_page_config(
        page_title="🎬 TV Show Recommendation Bot",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Инъекция CSS для кастомного оформления
    st.markdown("""
    <style>
        /* Основные стили */
        .stApp {
            background-color: #f5f5f5;
        }
        .stTextInput>div>div>input {
            border-radius: 12px;
            padding: 12px;
        }
        .stButton>button {
            border-radius: 12px;
            padding: 8px 16px;
            font-weight: 500;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            transform: scale(1.02);
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        /* Заголовки */
        h1, h2, h3, h4, h5, h6 {
            color: #4a4a4a;
        }
        
        /* Карточки */
        .stJson {
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Вкладки */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
            border-radius: 12px 12px 0 0;
            transition: all 0.3s;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4a4a4a;
            color: white;
        }
        
        /* Уведомления */
        .stAlert {
            border-radius: 12px;
        }
        
        /* Специальные классы */
        .cool-header {
            background: linear-gradient(45deg, #6a11cb 0%, #2575fc 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .response-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .status-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            margin-right: 8px;
        }
        .badge-info {
            background: #e6f3ff;
            color: #2575fc;
        }
        .badge-success {
            background: #e6f7e6;
            color: #2e7d32;
        }
        .badge-warning {
            background: #fff8e6;
            color: #ed6c02;
        }
        .badge-error {
            background: #ffebee;
            color: #d32f2f;
        }
    </style>
    """, unsafe_allow_html=True)

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
    st.markdown('<span class="status-badge badge-info">🔹 Инициализация Qdrant клиента</span>', unsafe_allow_html=True)
    st.markdown('<span class="status-badge badge-info">1. Проверяем наличие файла блокировки...</span>', unsafe_allow_html=True)
    lock_file = os.path.join(db_path, '.lock')
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
            st.markdown('<span class="status-badge badge-success">✅ Файл блокировки удален</span>', unsafe_allow_html=True)
        except OSError as e:
            st.markdown(f'<span class="status-badge badge-warning">⚠️ Ошибка удаления файла блокировки: {e}</span>', unsafe_allow_html=True)
    st.markdown('<span class="status-badge badge-info">2. Подключаемся к базе Qdrant...</span>', unsafe_allow_html=True)
    return qdrant_client.QdrantClient(path=db_path)

# Инициализируем клиенты с обработкой ошибок
try:
    client = initialize_qdrant_client(QDRANT_PATH)
    # Явно указываем device='cpu' для работы на Streamlit Cloud
    embedding_model = SentenceTransformer(MODEL_NAME, device='cpu')
    st.success("✅ Модели и клиенты успешно инициализированы")
except Exception as e:
    st.error(f"❌ Ошибка инициализации: {str(e)}")
    st.stop()

# --- Проверка API ключей ---
def check_api_keys():
    if not all([API_KEY, FOLDER_ID, YANDEX_TRANSLATE_API_KEY]):
        st.error("❌ Не все API ключи настроены! Проверьте secrets.toml")
        return False
    return True

# --- Функция определения русского текста ---
def is_russian(text):
    """Проверяет, содержит ли текст русские символы"""
    return bool(re.search('[а-яА-Я]', text))

# --- Функция перевода ---
def translate_text(text, target_lang="ru", source_lang=None):
    if not check_api_keys():
        return text
        
    st.markdown(f'<span class="status-badge badge-info">🔹 Запрос к Yandex Translate API ({source_lang or "auto"} -> {target_lang})</span>', unsafe_allow_html=True)
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
            st.markdown('<span class="status-badge badge-success">✅ Перевод успешно выполнен</span>', unsafe_allow_html=True)
            return response.json()["translations"][0]["text"]
        else:
            st.markdown(f'<span class="status-badge badge-error">❌ Ошибка API: {response.status_code}</span>', unsafe_allow_html=True)
            return text
    except Exception as e:
        st.markdown(f'<span class="status-badge badge-error">❌ Ошибка соединения: {str(e)}</span>', unsafe_allow_html=True)
        return text

# --- Поиск в Qdrant ---
def search_in_qdrant(query, top_k=3):
    try:
        st.markdown('<span class="status-badge badge-info">🔹 Векторизация запроса</span>', unsafe_allow_html=True)
        query_vector = embedding_model.encode(query, convert_to_tensor=True).cpu().numpy().tolist()
        
        st.markdown('<span class="status-badge badge-info">🔹 Поиск в Qdrant</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="status-badge badge-info">Ищем {top_k} ближайших соседей для запроса: \'{query}\'</span>', unsafe_allow_html=True)
        
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k
        )
        st.markdown(f'<span class="status-badge badge-success">✅ Найдено {len(search_result)} результатов</span>', unsafe_allow_html=True)
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
        
    st.markdown('<span class="status-badge badge-info">🔹 Формирование контекста для YandexGPT</span>', unsafe_allow_html=True)
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

    st.markdown('<span class="status-badge badge-info">🔹 Запрос к YandexGPT API</span>', unsafe_allow_html=True)
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
        st.markdown('<span class="status-badge badge-success">✅ Ответ от YandexGPT получен</span>', unsafe_allow_html=True)
        return response.json()['result']['alternatives'][0]['message']['text']
    except Exception as e:
        st.error(f"❌ Ошибка запроса к YandexGPT: {str(e)}")
        return f"Error: {str(e)}"

# --- Интерфейс Streamlit ---
def main():
    setup_styles()
    
    st.markdown('<div class="cool-header"><h1>🎬 TV Show Recommendation Bot</h1><p>Найдите идеальный сериал с помощью AI</p></div>', unsafe_allow_html=True)
    
    st.warning("⚠️ База данных на английском! Вводите запрос на английском (например: 'comedy about space'), но можешь написать на русском и посмотреть, что будет ;)")
    
    # Проверка даты ротации ключей
    if "last_key_rotation" not in st.session_state:
        st.session_state.last_key_rotation = datetime(2025, 9, 8)
    
    if (datetime.now() - st.session_state.last_key_rotation).days > 90:
        st.warning(f"🚨 Рекомендуется сменить API ключи! Последняя ротация: {st.session_state.last_key_rotation.strftime('%d.%m.%Y')}")

    tab1, tab2 = st.tabs(["🔍 Поиск сериалов", "🧪 Проверка RAG"])

    with tab1:
        st.markdown("### 🔍 Поиск по базе сериалов")
        st.markdown("Введите ваш запрос на английском (например: 'sci-fi series with aliens')")
        
        user_query = st.text_input("**Your query (in English):**", "recommend a series about space and aliens", key="query_input")

        if st.button("Search", key="search_btn", type="primary"):
            if not check_api_keys():
                st.error("Пожалуйста, настройте API ключи в secrets.toml")
                return
                
            with st.spinner("🔍 Обработка запроса..."):
                # Сохраняем информацию о языке запроса
                st.session_state.was_russian = is_russian(user_query)
                original_query = user_query
                
                # Если запрос на русском - переводим на английский
                if st.session_state.was_russian:
                    st.warning("Я не для того разбирался в Yandex API, чтобы вы обманывали систему! Ваш текст будет переведен на английский 😠")
                    user_query = translate_text(user_query, target_lang="en", source_lang="ru")
                    st.markdown(f'<div class="response-card"><p><strong>Переведенный запрос:</strong> {user_query}</p></div>', unsafe_allow_html=True)
                
                # Шаг 1: Поиск в Qdrant
                st.markdown("### 🔍 Найденные сериалы")
                shows = search_in_qdrant(user_query)
                if shows:
                    with st.expander("Показать сырые данные"):
                        st.json(shows, expanded=True)

                    # Шаг 2: Запрос к YandexGPT
                    st.markdown("### 🤖 Ответ AI")
                    gpt_response_en = ask_yandex_gpt(user_query, shows)
                    st.markdown(f'<div class="response-card"><p><strong>Ответ на английском:</strong></p><p>{gpt_response_en}</p></div>', unsafe_allow_html=True)
                    
                    # Сохраняем ответ для возможного перевода
                    st.session_state.gpt_response_en = gpt_response_en

                    # Шаг 3: Перевод на русский (если исходный запрос был на русском)
                    if st.session_state.was_russian:
                        st.markdown("### 🇷🇺 Перевод ответа")
                        gpt_response_ru = translate_text(gpt_response_en, target_lang="ru", source_lang="en")
                        st.markdown(f'<div class="response-card"><p><strong>Ответ на русском:</strong></p><p>{gpt_response_ru}</p></div>', unsafe_allow_html=True)
                        # Сохраняем перевод
                        st.session_state.gpt_response_ru = gpt_response_ru
                else:
                    st.warning("Не удалось найти сериалы по вашему запросу")
        
        # Добавляем кнопку перевода, если запрос был на английском и есть ответ
        if ('was_russian' in st.session_state and not st.session_state.was_russian and 
            'gpt_response_en' in st.session_state):
            if st.button("Перевести ответ на русский", key="translate_btn"):
                with st.spinner("🌍 Перевод ответа..."):
                    gpt_response_ru = translate_text(st.session_state.gpt_response_en, target_lang="ru", source_lang="en")
                    st.session_state.gpt_response_ru = gpt_response_ru
                    st.markdown("### 🇷🇺 Перевод ответа")
                    st.markdown(f'<div class="response-card"><p><strong>Ответ на русском:</strong></p><p>{gpt_response_ru}</p></div>', unsafe_allow_html=True)

    with tab2:
        st.markdown("### � Проверка RAG-системы")
        st.markdown("Здесь вы можете проверить, использует ли YandexGPT только предоставленный контекст")
        
        test_query = st.text_input("Тестовый запрос:", "What can you tell me about these shows?", key="test_query")
        
        if st.button("Проверить RAG", key="rag_btn", type="primary"):
            if not check_api_keys():
                st.error("Пожалуйста, настройте API ключи в secrets.toml")
                return
                
            with st.spinner("🔎 Анализ RAG-системы..."):
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
                    
                    st.markdown("### 🔧 Компоненты RAG")
                    with st.expander("System Prompt"):
                        st.markdown(f'<div class="response-card"><p>{system_prompt}</p></div>', unsafe_allow_html=True)
                    
                    with st.expander("Контекст из базы"):
                        st.markdown(f'<div class="response-card"><p>{context_str}</p></div>', unsafe_allow_html=True)
                    
                    with st.expander("Финальный промпт"):
                        st.markdown(f'<div class="response-card"><p>{final_prompt}</p></div>', unsafe_allow_html=True)
                    
                    st.success("Проверьте, что system prompt явно требует использовать только контекст!")
                else:
                    st.warning("Не удалось загрузить данные для проверки RAG")

if __name__ == "__main__":
    main()
