import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import os
from vector import get_retriever, start_indexing, check_ollama, DB_DIR

st.set_page_config(page_title="FIFA AI Pro", layout="wide")

# UI Header
st.title("⚽ FIFA World Cup Analyst")
st.markdown("---")

# 1. System Health Check
ollama_online = check_ollama()
db_exists = os.path.exists(DB_DIR)

col1, col2 = st.columns(2)
with col1:
    if ollama_online:
        st.success("✅ Ollama Server: Online")
    else:
        st.error("❌ Ollama Server: Offline. Run 'ollama serve' in your terminal.")

with col2:
    if db_exists:
        st.success("✅ Database: Ready")
    else:
        st.warning("⚠️ Database: Not Found")

# 2. Database Builder Logic
if not db_exists:
    st.info("The match database needs to be built from the CSV file. This takes 2-3 minutes.")
    if st.button("🚀 Start Building Database", disabled=not ollama_online):
        progress = st.progress(0, text="Starting...")
        try:
            start_indexing(progress)
            st.success("Database built successfully! Refreshing...")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")
    st.stop()

# 3. AI Chat Logic (Only runs if DB exists)
@st.cache_resource
def load_ai():
    retriever = get_retriever()
    llm = OllamaLLM(model="gemma3")
    prompt = ChatPromptTemplate.from_template("Match Data: {context}\n\nQuestion: {question}")
    return retriever, (prompt | llm)

if db_exists and ollama_online:
    retriever, chain = load_ai()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if user_query := st.chat_input("Ask about a World Cup match..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"): st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                docs = retriever.invoke(user_query)
                context = "\n".join([d.page_content for d in docs])
                response = chain.invoke({"context": context, "question": user_query})
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})