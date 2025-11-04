import streamlit as st
import json
import os
import hashlib

USER_DB_FILE = "user_db.json"


# ---------- Utility Functions ----------
def load_user_db():
    """Load user credentials from JSON file."""
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, "r") as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return {}
    return {}


def save_user_db(user_db):
    """Save user credentials to JSON file."""
    with open(USER_DB_FILE, "w") as file:
        json.dump(user_db, file, indent=4)


def hash_password(password: str) -> str:
    """Secure password hashing using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


# ---------- Login/Register UI ----------
def login_ui():
    st.title("🔐 Jublie Login Portal")

    user_db = load_user_db()

    # Default session states
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None

    option = st.radio("Select an option", ("Login", "Register"))

    if option == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_btn = st.button("Login")

        if login_btn:
            if username in user_db and user_db[username] == hash_password(password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.current_page = "law"
                st.success(f"Welcome back, {username} 👋")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    elif option == "Register":
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        register_btn = st.button("Register")

        if register_btn:
            if new_username in user_db:
                st.error("Username already exists. Please choose another.")
            elif not new_username or not new_password:
                st.error("Both fields are required.")
            else:
                user_db[new_username] = hash_password(new_password)
                save_user_db(user_db)
                st.success("Registration successful! You can now log in.")


# ---------- Logout ----------
def logout_button():
    """Logout and reset session state."""
    if st.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.current_page = "login"
        st.success("Logged out successfully!")
        st.rerun()
