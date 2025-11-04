import streamlit as st
from module.login import login_ui
from module.law_chatbot import show_law_chatbot
from module.general_chatbot import show_general_chatbot
from prediction_model import predict_judgment
import docx
import PyPDF2

# ---------------------
# Helper Functions
# ---------------------
def extract_text_from_file(uploaded_file):
    """Extracts text from TXT, DOCX, or PDF files."""
    try:
        if uploaded_file.type == "text/plain":
            return uploaded_file.read().decode("utf-8")

        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

        elif uploaded_file.type == "application/pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text.strip()

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
    return ""

# ---------------------
# Prediction UI
# ---------------------
def show_prediction_ui():
    """UI for hybrid ML + LLM legal prediction."""
    st.subheader("⚖️ Legal Judgment Prediction")

    scenario_text = st.text_area("📝 Enter scenario description", height=200)
    uploaded_file = st.file_uploader("📂 Or upload a case file (TXT, DOCX, PDF)", type=["txt", "docx", "pdf"])

    # Combine uploaded text with manual input
    if uploaded_file:
        file_text = extract_text_from_file(uploaded_file)
        if file_text:
            scenario_text = (scenario_text + "\n" + file_text).strip() if scenario_text else file_text

    if st.button("🔮 Predict Judgment"):
        if not scenario_text.strip():
            st.error("⚠️ Please enter a scenario or upload a file.")
            return

        with st.spinner("Analyzing scenario using ML + LLM..."):
            try:
                result = predict_judgment(scenario_text)

                st.success("✅ Analysis complete!")

                st.subheader("📜 Case Summary")
                st.write(result.get("summary", "No summary available."))

                st.subheader("🔮 Predicted Outcome")
                st.write(result.get("prediction", "No prediction available."))

                st.subheader("📊 Confidence")
                st.write(f"{result.get('confidence', 0)}%")

                st.subheader("📌 Reasons")
                for r in result.get("reasons", []):
                    st.markdown(f"- {r}")

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

# ---------------------
# Streamlit Configuration
# ---------------------
st.set_page_config(page_title="Jublie Chatbot", page_icon="⚖️", layout="wide")

# ---------------------
# Session State Initialization
# ---------------------
default_session = {
    "authenticated": False,
    "current_page": "login",
    "username": "",
    "law_messages": [],
    "general_messages": []
}
for key, default in default_session.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------
# Authentication
# ---------------------
if not st.session_state.authenticated:
    login_ui()
    st.stop()

# ---------------------
# Navigation
# ---------------------
st.sidebar.title("🌐 Navigation")
nav_choice = st.sidebar.radio(
    "Go to:",
    ("Law-Jublie ⚖️", "Com-Jublie 💬", "Logout 🚪"),
    index=0 if st.session_state.current_page == "law" else 1
)

if nav_choice.startswith("Law"):
    st.session_state.current_page = "law"
elif nav_choice.startswith("Com"):
    st.session_state.current_page = "general"
elif nav_choice.startswith("Logout"):
    st.session_state.clear()
    st.session_state["authenticated"] = False
    st.success("Logged out successfully!")
    st.rerun()

# ---------------------
# Page Rendering
# ---------------------
if st.session_state.current_page == "law":
    show_law_chatbot()
    st.markdown("---")
    show_prediction_ui()
elif st.session_state.current_page == "general":
    show_general_chatbot()
