import streamlit as st
from test_model import load_model, generate_response
import torch

# Page config
st.set_page_config(
    page_title="AI Dad Chat",
    page_icon="👨‍👦",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e6f3ff;
    }
    .dad-message {
        background-color: #f0f0f0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Load model
@st.cache_resource
def load_cached_model():
    with st.spinner("Loading AI Dad... This might take a minute."):
        model, tokenizer = load_model()
    return model, tokenizer

# Header
st.title("💭 Chat with AI Dad")
st.markdown("---")

# Load the model
model, tokenizer = load_cached_model()

# Chat interface
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    
    with st.container():
        if role == "user":
            st.markdown(f"""
                <div class="chat-message user-message">
                    <div><strong>You:</strong> {content}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="chat-message dad-message">
                    <div><strong>Dad:</strong> {content}</div>
                </div>
            """, unsafe_allow_html=True)

# Chat input
with st.container():
    user_input = st.text_input("Type your message:", key="user_input", placeholder="What's on your mind, son?")
    
    if st.button("Send", key="send") or user_input:
        if user_input:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Generate Dad's response
            with st.spinner("Dad is typing..."):
                response = generate_response(user_input, model, tokenizer)
            
            # Add Dad's response to chat
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Clear input
            st.rerun()

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Footer
st.markdown("---")
st.markdown("💡 _Feel free to share anything with your AI Dad. He's here to listen and support you._") 