import streamlit as st
import requests
import qdrant_client
from sentence_transformers import SentenceTransformer
import os
from datetime import datetime
import re
import torch
import time
from random import choice

# --- Настройка темной темы ---
def setup_dark_theme():
    st.set_page_config(
        page_title="🎬 TV Show Recommendation Bot",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Инъекция CSS для темной темы
    st.markdown("""
    <style>
        /* Основные стили */
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
        }
        
        /* Заголовки */
        h1, h2, h3, h4, h5, h6 {
            color: #FFFFFF;
            font-weight: 600;
            margin-bottom: 0.75rem;
        }
        
        /* Основной текст */
        p, div, span, label {
            color: #E0E0E0 !important;
        }
        
        /* Текстовые поля ввода */
        .stTextInput>div>div>input {
            border: 1px solid #444;
            border-radius: 8px;
            padding: 10px 12px;
            font-size: 14px;
            color: #FFFFFF;
            background-color: #1E1E1E;
        }
        
        /* Кнопки */
        .stButton>button {
            background-color: #1DB954;
            color: #000000 !important;
            border: none;
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 500;
            font-size: 14px;
            transition: all 0.2s;
        }
        .stButton>button:hover {
            background-color: #1ED760;
            transform: translateY(-1px);
        }
        
        /* Карточки */
        .card {
            background: #1E1E1E;
            border-radius: 8px;
            border: 1px solid #333;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Вкладки */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            padding: 0;
            border-bottom: none;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 16px;
            border-radius: 8px;
            background-color: #333;
            color: #E0E0E0 !important;
            border: none;
            margin: 0;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1DB954;
            color: #000000 !important;
        }
        
        /* Уведомления */
        .stAlert {
            border-radius: 8px;
            padding: 12px 16px;
            border: 1px solid #444;
        }
        .stAlert [data-testid="stMarkdownContainer"] {
            color: inherit !important;
        }
        
        /* JSON viewer */
        .stJson {
            border-radius: 8px;
            border: 1px solid #444;
            background-color: #1E1E1E !important;
        }
        
        /* Прогресс-бар */
        .stProgress>div>div>div {
            background-color: #1DB954;
        }
        
        /* Выпадающие списки */
        .stSelectbox>div>div>div {
            color: #FFFFFF;
            background-color: #1E1E1E;
        }
        
        /* Ховер-эффекты */
        .stButton>button:hover, .stTabs [data-baseweb="tab"]:hover {
            opacity: 0.9;
        }
        
        /* Анимации */
        @keyframes shake {
            0%, 100% {transform: translateX(0);}
            10%, 30%, 50%, 70%, 90% {transform: translateX(-5px);}
            20%, 40%, 60%, 80% {transform: translateX(5px);}
        }
        .shake {
            animation: shake 0.5s;
        }
        @keyframes rainbow {
            0% {color: red;}
            14% {color: orange;}
            28% {color: yellow;}
            42% {color: green;}
            57% {color: blue;}
            71% {color: indigo;}
            85% {color: violet;}
            100% {color: red;}
        }
        .rainbow {
            animation: rainbow 2s linear infinite;
        }
        @keyframes spin {
            0% {transform: rotate(0deg);}
            100% {transform: rotate(360deg);}
        }
        .spin {
            display: inline-block;
            animation: spin 1s linear infinite;
        }
    </style>
    """, unsafe_allow_html=True)

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
    st.info("🔹 **Инициализация Qdrant клиента**")
    st.info("1. Проверяем наличие файла блокировки...")
    lock_file = os.path.join(db_path, '.lock')
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
            st.success("✅ Файл блокировки удален")
        except OSError as e:
            st.warning(f"⚠️ Ошибка удаления файла блокировки: {e}")
    st.info("2. Подключаемся к базе Qdrant...")
    return qdrant_client.QdrantClient(path=db_path)

try:
    client = initialize_qdrant_client(QDRANT_PATH)
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
    return bool(re.search('[а-яА-Я]', text))

# --- Функция перевода ---
def translate_text(text, target_lang="ru", source_lang=None):
    if not check_api_keys():
        return text
        
    st.info(f"🔹 **Запрос к Yandex Translate API ({source_lang or 'auto'} -> {target_lang})**")
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
            st.success("✅ Перевод успешно выполнен")
            return response.json()["translations"][0]["text"]
        else:
            st.error(f"❌ Ошибка API: {response.status_code}")
            return text
    except Exception as e:
        st.error(f"❌ Ошибка соединения: {str(e)}")
        return text

# --- Поиск в Qdrant ---
def search_in_qdrant(query, top_k=3):
    try:
        st.info("🔹 **Векторизация запроса**")
        query_vector = embedding_model.encode(query, convert_to_tensor=True).cpu().numpy().tolist()
        
        st.info("🔹 **Поиск в Qdrant**")
        st.info(f"Ищем {top_k} ближайших соседей для запроса: '{query}'")
        
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k
        )
        st.success(f"✅ Найдено {len(search_result)} результатов")
        return [hit.payload for hit in search_result]
    except Exception as e:
        st.error(f"❌ Ошибка поиска в Qdrant: {str(e)}")
        return []

# --- Запрос к YandexGPT ---
def ask_yandex_gpt(user_query, context, check_rag=False):
    if not context:
        return "According to our TV shows database: No relevant shows found."
    
    if not check_api_keys():
        return "API keys not configured"
        
    st.info("🔹 **Формирование контекста для YandexGPT**")
    context_str = "\n".join([
        f"- Title: {show.get('title', 'N/A')}, Genres: {show.get('genres', 'N/A')}, Description: {show.get('description', 'N/A')}" 
        for show in context
    ])

    system_prompt = """You are a TV show recommendation assistant working with our TV shows database. 
    Always begin your response with: "According to our TV shows database:"
    Then provide recommendations based ONLY on the context provided below. 
    If you don't know the answer, say 'I don't have enough information in our database'.
    
    Important rules:
    1. Never mention that you're an AI assistant
    2. Always refer to "our database" when providing information
    3. Keep responses concise but informative
    4. If multiple shows match, list them with brief descriptions
    
    Context:
    """ + context_str
    
    final_prompt = f"User question: {user_query}"

    if check_rag:
        return system_prompt, final_prompt, context_str

    st.info("🔹 **Запрос к YandexGPT API**")
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
        st.success("✅ Ответ от YandexGPT получен")
        return response.json()['result']['alternatives'][0]['message']['text']
    except Exception as e:
        st.error(f"❌ Ошибка запроса к YandexGPT: {str(e)}")
        return f"Error: {str(e)}"

# --- Веселый эффект при русском запросе ---
def show_funny_effect():
    effects = [
        lambda: st.markdown('<div class="shake">🤪</div>', unsafe_allow_html=True),
        lambda: st.markdown('<div class="rainbow">🌈 Ну давай, не читай рекомендации, потрать мои миллион токенов на перевод! 🌈</div>', unsafe_allow_html=True),
        lambda: st.markdown('<div class="spin">🌀</div> Ну давай, не читай рекомендации, потрать мои миллион токенов на перевод!', unsafe_allow_html=True),
        lambda: st.balloons(),
        lambda: st.snow(),
        lambda: st.error("💸 Токены горят!"),
        lambda: st.warning("⚠️ Внимание! Перевод стоит денег!"),
        lambda: st.success("🎉 Ура! Еще один перевод!"),
    ]
    
    chosen_effect = choice(effects)
    chosen_effect()
    
    # Добавляем несколько эффектов сразу для большего веселья
    if choice([True, False]):
        time.sleep(0.5)
        choice(effects)()

# --- Интерфейс Streamlit ---
def main():
    setup_dark_theme()
    
    st.title("🎬 TV Show Recommendation Bot")
    st.markdown("""
    <div class="card">
        <p>Найдите идеальный сериал с помощью AI. База данных на английском - вводите запрос на английском для лучших результатов, или введи на русском и посмотри что будет 😈</p>
    </div>
    """, unsafe_allow_html=True)
    
    if "last_key_rotation" not in st.session_state:
        st.session_state.last_key_rotation = datetime(2025, 9, 8)
    
    if (datetime.now() - st.session_state.last_key_rotation).days > 90:
        st.warning(f"🚨 Рекомендуется сменить API ключи! Последняя ротация: {st.session_state.last_key_rotation.strftime('%d.%m.%Y')}")

    tab1, tab2 = st.tabs(["🔍 Поиск сериалов", "🧪 Проверка RAG"])

    with tab1:
        st.markdown("### Поиск по базе сериалов")
        st.markdown("Введите ваш запрос на английском (например: 'sci-fi series with aliens') или на русском (но мы предупредили!)")
        
        user_query = st.text_input("**Ваш запрос:**", "recommend a series about space and aliens", key="query_input")

        if st.button("Поиск", key="search_btn"):
            if not check_api_keys():
                st.error("Пожалуйста, настройте API ключи в secrets.toml")
                return
                
            with st.spinner("Обработка запроса..."):
                st.session_state.was_russian = is_russian(user_query)
                original_query = user_query
                
                if st.session_state.was_russian:
                    show_funny_effect()
                    st.markdown("""
                    <div class="card" style="border: 2px solid #ff0000;">
                        <p style="color: #ff0000; font-weight: bold;">Ну давай, не читай рекомендации, потрать мои миллион токенов на перевод! 😭</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    user_query = translate_text(user_query, target_lang="en", source_lang="ru")
                    st.markdown(f"""
                    <div class="card">
                        <p><strong>Переведенный запрос:</strong> {user_query}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("### Найденные сериалы")
                shows = search_in_qdrant(user_query)
                if shows:
                    with st.expander("Показать сырые данные"):
                        st.json(shows, expanded=True)

                    st.markdown("### Ответ AI")
                    gpt_response_en = ask_yandex_gpt(user_query, shows)
                    st.markdown(f"""
                    <div class="card">
                        <p><strong>Ответ на английском:</strong></p>
                        <p>{gpt_response_en}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.session_state.gpt_response_en = gpt_response_en

                    if st.session_state.was_russian:
                        st.markdown("### Перевод ответа")
                        gpt_response_ru = translate_text(gpt_response_en, target_lang="ru", source_lang="en")
                        st.markdown(f"""
                        <div class="card">
                            <p><strong>Ответ на русском:</strong></p>
                            <p>{gpt_response_ru}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.session_state.gpt_response_ru = gpt_response_ru
                else:
                    st.warning("Не удалось найти сериалы по вашему запросу")
        
        if ('was_russian' in st.session_state and not st.session_state.was_russian and 
            'gpt_response_en' in st.session_state):
            if st.button("Перевести ответ на русский", key="translate_btn"):
                with st.spinner("Перевод ответа..."):
                    show_funny_effect()
                    gpt_response_ru = translate_text(st.session_state.gpt_response_en, target_lang="ru", source_lang="en")
                    st.session_state.gpt_response_ru = gpt_response_ru
                    st.markdown("### Перевод ответа")
                    st.markdown(f"""
                    <div class="card">
                        <p><strong>Ответ на русском:</strong></p>
                        <p>{gpt_response_ru}</p>
                    </div>
                    """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### Проверка RAG-системы")
        st.markdown("Проверьте, использует ли модель только предоставленный контекст")
        
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
                    
                    st.markdown("### Компоненты RAG")
                    with st.expander("System Prompt"):
                        st.markdown(f"""
                        <div class="card">
                            <p>{system_prompt}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with st.expander("Контекст из базы"):
                        st.markdown(f"""
                        <div class="card">
                            <p>{context_str}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with st.expander("Финальный промпт"):
                        st.markdown(f"""
                        <div class="card">
                            <p>{final_prompt}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.success("Проверьте, что system prompt явно требует использовать только контекст!")
                else:
                    st.warning("Не удалось загрузить данные для проверки RAG")

if __name__ == "__main__":
    main()
