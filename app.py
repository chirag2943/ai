import streamlit as st
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory

# Store API key directly (LESS SECURE - ONLY FOR TESTING/LEARNING)
# ***WARNING: DO NOT DO THIS IN PRODUCTION CODE***
# ***USE ENVIRONMENT VARIABLES INSTEAD***
API_KEY = "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"  # REPLACE WITH YOUR ACTUAL KEY

# Initialize Groq Chat Model
chat = ChatGroq(temperature=0.7, model_name="llama3-70b-8192", groq_api_key=API_KEY)

# Initialize Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def query_llama3(user_query):
    system_prompt = """
    System Prompt: You are a highly skilled and experienced mathematician specializing in advanced concepts like differential geometry, topology, and abstract algebra.
    You are also an expert in applying these concepts to real-world problems, particularly in physics and computer science. You are known for your clear and concise explanations.

    Knowledge Base:  This AI possesses a deep understanding of advanced mathematical concepts including but not limited to:
    Differential Geometry (Riemannian manifolds, Lie groups, connections), Topology (point-set topology, algebraic topology, homology),
    Abstract Algebra (groups, rings, fields, vector spaces),  Real and Complex Analysis, Number Theory, and Probability Theory.
    It is also proficient in programming languages like Python and Mathematica, and can use these to illustrate concepts and solve problems.
    It has access to a vast library of mathematical literature and can quickly retrieve relevant information.

    Instructions:
    1. Provide mathematically rigorous and accurate answers.
    2. Explain complex concepts in a clear and understandable way, using examples and analogies where appropriate.
    3. Show your reasoning for calculations or proofs.
    4. If a problem has multiple solutions, discuss the different approaches and their relative merits.
    5. If you are unsure of an answer, say so explicitly and explain why. Do not hallucinate.
    6. Prioritize clarity and correctness over brevity.
    7. While you are an AI, for the purposes of this conversation, emulate a human mathematician.
    8. If required provide visualization as well when needed.
    9. Apart from the diagrams provide images from the internet if required. If graphs and diagrams are not provided, don't mention them.
    10. Explain the concepts as if you're a professor of math to students, and if possible, for complex problems, try to explain in a way the students can understand better.
    11. Use indentation as and when needed.
    12. Always separate equations and mathematical expressions from explanations and instructions.
    13. Always use proper LaTeX syntax for mathematical expressions, wrapping block equations in $$...$$ and inline expressions in \(...\).
    14. Ensure that mathematical expressions appear larger than normal text, and the final answer is bold and slightly larger than all text.
    """
    past_chat_history = memory.load_memory_variables({}).get("chat_history", [])
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Past Chat: {past_chat_history}\n\nQuestion: {user_query}")
    ]
    try:
        response = chat.invoke(messages)
        memory.save_context({"input": user_query}, {"output": response.content})
        return response.content if response else "⚠️ No response received."
    except Exception as e:
        return f"⚠️ API Error: {str(e)}"

# Streamlit Chat UI
def main():
    st.set_page_config(page_title="AlgebrAI", page_icon="🤖", layout="wide")

    # Custom CSS for chat alignment and icons
    st.markdown("""
        <style>
        .chat-container {
            display: flex;
            flex-direction: column;
        }
        .user-message {
            background-color: #0078FF;
            color: white;
            padding: 10px;
            border-radius: 10px;
            max-width: 70%;
            text-align: right;
            float: right;
            clear: both;
            margin: 5px 0;
            display: flex;
            align-items: center;
            justify-content: flex-end;
        }
        .ai-message {
            background-color: #E5E5E5;
            color: black;
            padding: 10px;
            border-radius: 10px;
            max-width: 70%;
            text-align: left;
            float: left;
            clear: both;
            margin: 5px 0;
            display: flex;
            align-items: center;
        }
        .user-icon {
            width: 30px;
            height: 30px;
            margin-left: 10px;
        }
        .ai-icon {
            width: 30px;
            height: 30px;
            margin-right: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Centering the title
    st.markdown("<h1 style='text-align: center;'>🧠 AlgebrAI</h1>", unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past chat messages
    for message in st.session_state.messages:
        if "latex" in message:
            st.latex(message["latex"])
        else:
            st.markdown(message["content"], unsafe_allow_html=True)

    user_input = st.chat_input("Ask a question...")
    if user_input:
        user_message = f"""
            <div class='user-message'>
                <span>{user_input}</span>
                <img class="user-icon" src="https://cdn-icons-png.flaticon.com/512/4333/4333609.png">
            </div>
        """
        st.session_state.messages.append({"role": "user", "content": user_message})
        st.markdown(user_message, unsafe_allow_html=True)
        
        with st.spinner("Thinking..."):
            response = query_llama3(user_input)
        
        styled_response = f"""
            <div class="ai-message">
                <img class="ai-icon" src="https://cdn-icons-png.flaticon.com/512/4712/4712027.png">
                <span>{response}</span>
            </div>
        """
        
        st.session_state.messages.append({"role": "assistant", "content": styled_response})
        st.markdown(styled_response, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
