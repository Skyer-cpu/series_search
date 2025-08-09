import streamlit as st
import requests
import qdrant_client
from sentence_transformers import SentenceTransformer
import os
from datetime import datetime

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
    st.write("🔹 Инициализация Qdrant клиента")
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
        
    st.write("🔹 Запрос к Yandex Translate API")
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
    st.write("🔹 Векторизация запроса")
    query_vector = embedding_model.encode(query).tolist()
    
    st.write(f"🔹 Поиск в Qdrant (топ-{top_k} для '{query}')")
    
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
        
    st.write("🔹 Формирование контекста для YandexGPT")
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

    st.write("🔹 Запрос к YandexGPT API")
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

# --- Функция для вкладки архитектуры ---
def show_system_architecture():
    # Автоматическое определение темы
    is_dark = st._config.get("theme", {}).get("base") == "dark"
    
    # Динамические стили
    theme_css = f"""
    <style>
        .component-card {{
            background: {"#2a2a2a" if is_dark else "white"};
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border-left: 4px solid {"#4CAF50" if is_dark else "#2e7d32"};
            color: {"white" if is_dark else "#333333"};
        }}
        .flow-arrow {{
            text-align: center;
            margin: 5px 0;
            color: {"#64B5F6" if is_dark else "#2196F3"};
            font-size: 24px;
        }}
        .diagram-box {{
            background: {"#333333" if is_dark else "#f8f9fa"};
            padding: 15px;
            border-radius: 10px;
            margin-top: 30px;
            color: {"white" if is_dark else "#333333"};
        }}
        pre {{
            background: {"#1e1e1e" if is_dark else "white"} !important;
            color: {"#f0f0f0" if is_dark else "#333333"} !important;
            border: 1px solid {"#555" if is_dark else "#ddd"} !important;
        }}
    </style>
    """
    st.markdown(theme_css, unsafe_allow_html=True)

    st.header("🔧 Архитектура RAG-системы")
    
    # Компоненты системы
    components = [
        {"title": "1. Пользовательский ввод", "desc": "Запрос на английском (например: 'comedy about space')"},
        {"title": "2. Векторизация запроса", "desc": "Модель SentenceTransformer преобразует текст в вектор"},
        {"title": "3. Поиск в Qdrant", "desc": "Поиск сериалов по векторному сходству"},
        {"title": "4. Формирование контекста", "desc": "Подготовка данных для LLM"},
        {"title": "5. Генерация ответа (YandexGPT)", "desc": "Строго по предоставленному контексту"},
        {"title": "6. Перевод ответа", "desc": "Yandex Translate API → русский язык"},
        {"title": "7. Вывод результата", "desc": "Персонализированные рекомендации"}
    ]

    for comp in components:
        st.markdown(f"""
        <div class="component-card">
            <h3>{comp['title']}</h3>
            <p>{comp['desc']}</p>
        </div>
        """, unsafe_allow_html=True)
        if comp != components[-1]:
            st.markdown('<div class="flow-arrow">↓</div>', unsafe_allow_html=True)

    # Диаграмма
    st.markdown("""
    <div class="diagram-box">
        <h3>📌 Диаграмма последовательности</h3>
        <pre>
Пользователь → Streamlit → Векторизация → Qdrant → Формирование контекста → YandexGPT → Переводчик → Пользователь
        </pre>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📚 Подробнее о RAG", expanded=False):
        st.markdown("""
        **Retrieval-Augmented Generation (RAG)** сочетает:
        - Поиск в базе знаний
        - Генерацию ответа с привязкой к контексту
        
        **Преимущества:**
        - Нет "галлюцинаций" модели
        - Актуальные данные из базы
        - Поддержка двуязычных запросов
        """)

    with st.expander("🛠 Технологии", expanded=False):
        cols = st.columns(2)
        with cols[0]:
            st.markdown("""
            **Основные компоненты:**
            - 🗄️ Qdrant
            - 🤖 Sentence Transformers
            - 🧠 YandexGPT API
            - 🌍 Yandex Translate API
            """)
        with cols[1]:
            st.markdown("""
            **Инфраструктура:**
            - 🐍 Python 3.10+
            - 🚀 Streamlit
            - 🔐 Yandex Cloud
            """)

# --- Основной интерфейс ---
def main():
    st.title("🎬 TV Show Recommendation Bot")
    st.warning("⚠️ База на английском! Вводите запрос на английском (например: 'comedy about space')")
    
    if "last_key_rotation" not in st.session_state:
        st.session_state.last_key_rotation = datetime(2025, 9, 8)
    
    if (datetime.now() - st.session_state.last_key_rotation).days > 90:
        st.warning(f"🚨 Рекомендуется сменить API ключи! Последняя ротация: {st.session_state.last_key_rotation.strftime('%d.%m.%Y')}")

    tab1, tab2, tab3 = st.tabs(["🔍 Поиск сериалов", "🧪 Проверка RAG", "🏗️ Архитектура"])

    with tab1:
        user_query = st.text_input("**Your query (in English):**", "recommend a series about space and aliens")
        if st.button("Search"):
            if not check_api_keys():
                return
                
            with st.spinner("Обработка..."):
                shows = search_in_qdrant(user_query)
                if shows:
                    st.subheader("🔍 Результаты поиска")
                    st.json(shows, expanded=True)

                    st.subheader("🤖 Ответ YandexGPT")
                    gpt_response = ask_yandex_gpt(user_query, shows)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.text_area("Английский", gpt_response, height=200)
                    with col2:
                        st.text_area("Русский", translate_text(gpt_response), height=200)
                else:
                    st.warning("Ничего не найдено")

    with tab2:
        st.subheader("Проверка RAG")
        test_query = st.text_input("Тестовый запрос:", "What can you tell me about these shows?")
        if st.button("Проверить"):
            if not check_api_keys():
                return
                
            with st.spinner("Анализ..."):
                shows = search_in_qdrant("comedy series", top_k=2)
                if shows:
                    system_prompt, final_prompt, context = ask_yandex_gpt(test_query, shows, True)
                    
                    st.subheader("🔧 Компоненты RAG")
                    with st.expander("System Prompt"):
                        st.code(system_prompt, language="text")
                    with st.expander("Контекст"):
                        st.code(context, language="text")
                    with st.expander("Полный промпт"):
                        st.code(final_prompt, language="text")
                else:
                    st.warning("Ошибка загрузки данных")

    with tab3:
        show_system_architecture()

if __name__ == "__main__":
    main()