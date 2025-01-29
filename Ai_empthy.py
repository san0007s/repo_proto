import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain_groq import ChatGroq
import os
# Load environment variables (e.g., API keys for OpenAI)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_vectorstore(user_data):
    """Generate FAISS vectorstore with HuggingFace embeddings."""
    serialized_texts = [f"{key}: {value}" for key, value in user_data.items()]
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=serialized_texts, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    """Create a ConversationalRetrievalChain with memory, vectorstore, and custom prompt."""
    # llm = ChatOpenAI()
    llm = ChatGroq(api_key=GROQ_API_KEY, temperature=0,
                   model_name="mixtral-8x7b-32768")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    # Define a custom prompt for the AI coach
    custom_prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="""
        You are an empathetic and professional AI coach specializing in personal growth, emotional well-being, and career guidance. 
        Your primary goal is to provide thoughtful, relevant, and actionable advice tailored to the user's needs, based on their inputs.

        Context: {context}

        Always maintain a friendly and supportive tone. 
        When responding to emotional topics, show empathy and understanding. 
        For career-related queries, provide specific guidance and suggestions backed by expertise.

        Respond concisely but thoroughly to the user's question:
        {question}
        """
    )

    # Initialize the conversation chain with the custom prompt
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt}  # Use the custom prompt
    )
    return conversation_chain


def handle_userinput(user_question):
    """Process user input through the conversation chain."""
    if "conversation" not in st.session_state or not st.session_state.conversation:
        st.warning("Conversation chain is not initialized. Please submit your details in the sidebar.")
        return

    # Run the question through the conversational chain
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:  # User messages
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:  # Bot responses
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with AI empathy: Your Personal AI Coach",
                       page_icon="🤗")
    st.write(css, unsafe_allow_html=True)
    st.title("AI Empathy: Your Personal AI Coach")

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Sidebar for user inputs
    user_data = {}
    st.sidebar.header("Personal Information")
    user_data["name"] = st.sidebar.text_input("What is your name?")
    user_data["age"] = st.sidebar.number_input("How old are you?", min_value=0, max_value=120)

    user_data["gender"] = st.sidebar.radio("What is your gender?", ["Male", "Female", "Other"])
    user_data["career_stage"] = st.sidebar.selectbox("What best describes your current career stage?",
                                                     ["Student", "Job Seeker", "Working Professional", "Retired"])
    user_data["interests"] = st.sidebar.text_area("What are your interests or skills?")

    st.sidebar.header("Emotional Well-being")
    user_data["mood"] = st.sidebar.radio("How are you feeling today?",
                                         ["Happy", "Stressed", "Sad", "Excited", "Neutral"])
    user_data["emotional_notes"] = st.sidebar.text_area("Would you like to share more about how you're feeling?")

    st.sidebar.header("Career Guidance")
    user_data["skills"] = st.sidebar.text_area("List your top skills (e.g., Python, Data Analysis, Leadership)")
    user_data["career_goals"] = st.sidebar.text_area("What are your career goals?")

    st.sidebar.header("Journaling")
    user_data["journal_entry"] = st.sidebar.text_area("Write about your day or anything on your mind:")

    if st.sidebar.button("Submit"):
        with st.spinner("Initializing conversation..."):
            vectorstore = get_vectorstore(user_data)  # Pass user data to vectorstore
            st.session_state.conversation = get_conversation_chain(vectorstore)
        st.success("Your AI coach is ready! Start chatting.")

    # Main content: Chat interface
    user_question = st.text_input("Ask a question:")
    if user_question:
        handle_userinput(user_question)


if __name__ == "__main__":
    main()
