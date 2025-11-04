import streamlit as st
import re
import os
import random
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from groq import Groq

# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
LLM_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found in .env file. Please set it before running.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ------------------------------
# Fun facts database
# ------------------------------
FUN_FACTS = [
    "🍯 Honey never spoils! Archaeologists found edible honey over 3,000 years old in Egyptian tombs.",
    "🐙 Octopuses have three hearts — two for the gills, one for the body.",
    "🍌 Bananas are berries, but strawberries aren't!",
    "🌍 A day on Venus is longer than a year on Venus.",
    "🦈 Sharks existed before trees."
]

# ------------------------------
# General Chatbot UI
# ------------------------------
def show_general_chatbot():
    st.subheader("🤖 General Chatbot — Jublie")

    # Language options
    languages = {
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
        "Hindi": "hi",
        "Japanese": "ja",
        "Chinese (Simplified)": "zh-CN",
        "Russian": "ru",
    }

    target_language_name = st.selectbox("🌐 Translate responses into:", list(languages.keys()))
    target_language = languages[target_language_name]

    # Initialize chat history
    if "general_messages" not in st.session_state:
        st.session_state.general_messages = []

    # Display chat history
    for msg in st.session_state.general_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    # User input
    general_query = st.chat_input("Ask anything...")
    if not general_query:
        return

    # Save user message
    st.session_state.general_messages.append({"role": "user", "content": general_query})
    with st.chat_message("user"):
        st.markdown(general_query)

    question = general_query.strip().lower()
    bot_reply = ""

    # ---------------- Simple Rules ----------------
    name_patterns = [
        r"what('?s| is) your name\??",
        r"who are you\??",
        r"tell me your name",
        r"your name\??",
        r"may i know your name\??"
    ]

    if any(re.fullmatch(p, question) for p in name_patterns):
        bot_reply = "My name is **Jublie**, your friendly assistant 🤖"
    elif "your role" in question or "what do you do" in question:
        bot_reply = "I’m a helpful assistant here to answer your questions and make your day easier!"
    elif "fun fact" in question or "trivia" in question:
        bot_reply = random.choice(FUN_FACTS)
    elif "news" in question:
        bot_reply = "📰 Sorry, live news fetching isn’t available right now."
    elif "translate" in question:
        try:
            translated_query = GoogleTranslator(source="auto", target=target_language).translate(general_query)
            bot_reply = f"**Translated to {target_language_name}:** {translated_query}"
        except Exception as e:
            bot_reply = f"⚠️ Translation error: {e}"
    else:
        # ---------------- Groq API LLM ----------------
        with st.spinner("💭 Jublie is thinking..."):
            try:
                system_prompt = (
                    "You are a friendly AI assistant named **Jublie**.\n"
                    "If asked your name, reply only with 'My name is Jublie.'\n"
                    "If asked your role, say you're a helpful AI assistant.\n"
                    "For Article queries (like Article 40, 47, 102, etc.), give a short 2–3 line summary only.\n"
                    "Ignore anything about Meta, LLaMA, Groq, or model details.\n"
                    "Always reply naturally and politely."
                )

                completion = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "system", "content": system_prompt}]
                            + [{"role": msg["role"], "content": msg["content"]}
                               for msg in st.session_state.general_messages],
                    temperature=0.3,
                    max_tokens=512
                )

                bot_reply = completion.choices[0].message.content.strip()

                # Filter backend mentions
                if re.search(r"(llama|meta|groq|model)", bot_reply, re.IGNORECASE):
                    bot_reply = "My name is Jublie."

                # Format code blocks if detected
                if bot_reply.startswith("#include"):
                    bot_reply = f"```c\n{bot_reply}\n```"

            except Exception as e:
                msg = str(e).lower()
                if "model_not_found" in msg or "does not exist" in msg:
                    bot_reply = (
                        "⚠️ The configured Groq model wasn’t found.\n"
                        "Automatically switching to `llama-3.1-70b-versatile`."
                    )
                    # ✅ Fallback model
                    try:
                        completion = client.chat.completions.create(
                            model="llama-3.1-70b-versatile",
                            messages=[{"role": "system", "content": system_prompt}]
                                    + [{"role": msg["role"], "content": msg["content"]}
                                       for msg in st.session_state.general_messages],
                            temperature=0.3,
                            max_tokens=512
                        )
                        bot_reply = completion.choices[0].message.content.strip()
                    except Exception as inner_e:
                        bot_reply = f"❌ Model error after fallback: {inner_e}"
                elif "rate_limit" in msg:
                    bot_reply = "⚠️ Groq API rate limit reached. Try again later."
                elif "invalid_api_key" in msg:
                    bot_reply = "❌ Invalid Groq API key. Check your .env file."
                elif "model_decommissioned" in msg:
                    bot_reply = "❌ Model decommissioned. Update to a supported model."
                else:
                    bot_reply = f"⚠️ Unexpected Groq API error: {e}"

    # Save assistant message
    st.session_state.general_messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply, unsafe_allow_html=True)

    st.markdown("---")
