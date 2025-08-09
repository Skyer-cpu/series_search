# ... (остальной код остается без изменений до функции main())

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
                # Сохраняем информацию о языке запроса
                st.session_state.was_russian = is_russian(user_query)
                original_query = user_query
                
                # Если запрос на русском - переводим на английский
                if st.session_state.was_russian:
                    st.warning("Я не для того разбирался в Yandex API, чтобы вы обманывали систему! Ваш текст будет переведен на английский 😠")
                    user_query = translate_text(user_query, target_lang="en", source_lang="ru")
                    st.write(f"Переведенный запрос: {user_query}")
                
                # Шаг 1: Поиск в Qdrant
                st.subheader("🔍 Найденные сериалы (сырые данные)")
                shows = search_in_qdrant(user_query)
                if shows:
                    st.json(shows, expanded=True)

                    # Шаг 2: Запрос к YandexGPT
                    st.subheader("🤖 Ответ YandexGPT")
                    gpt_response_en = ask_yandex_gpt(user_query, shows)
                    st.text_area("Ответ на английском", gpt_response_en, height=200)
                    
                    # Сохраняем ответ для возможного перевода
                    st.session_state.gpt_response_en = gpt_response_en

                    # Шаг 3: Перевод на русский (если исходный запрос был на русском)
                    if st.session_state.was_russian:
                        st.subheader("🇷🇺 Перевод ответа")
                        gpt_response_ru = translate_text(gpt_response_en, target_lang="ru", source_lang="en")
                        st.text_area("Ответ на русском", gpt_response_ru, height=200)
                        # Сохраняем перевод
                        st.session_state.gpt_response_ru = gpt_response_ru
                else:
                    st.warning("Не удалось найти сериалы по вашему запросу")
        
        # Добавляем кнопку перевода, если запрос был на английском и есть ответ
        if ('was_russian' in st.session_state and not st.session_state.was_russian and 
            'gpt_response_en' in st.session_state):
            if st.button("Перевести ответ на русский", key="translate_btn"):
                with st.spinner("Перевод ответа..."):
                    gpt_response_ru = translate_text(st.session_state.gpt_response_en, target_lang="ru", source_lang="en")
                    st.session_state.gpt_response_ru = gpt_response_ru
                    st.subheader("🇷🇺 Перевод ответа")
                    st.text_area("Ответ на русском", gpt_response_ru, height=200)

    with tab2:
        # ... (остальной код вкладки RAG остается без изменений)

if __name__ == "__main__":
    main()
