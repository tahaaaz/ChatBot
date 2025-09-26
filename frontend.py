import streamlit as st
from chatbot import Chatbot
import random
import time
def main():


    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #89f7fe, #66a6ff, #ff9a9e, #fad0c4);
            background-size: 400% 400%;
            animation: gradientBG 5s ease infinite;
        }

        @keyframes gradientBG {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🤖 Lambdatest Chatbot ")
    st.write("Smarter, faster, and learns from you.")

    bot = Chatbot("queries.json")
    
    # define clear function
    def clear_input():
        
        st.session_state["user_input"] = ""

    # Input + Clear button inline
    col1, col2 = st.columns([4, 1])
    with col1:
        user_query = st.text_input("You:", key="user_input", label_visibility="collapsed")

    with col2:
        st.button("❌ Clear", on_click=clear_input)

    if user_query:
        responses, best_key, score = bot.match(user_query)
        startt=time.time()
        if responses:
            st.success(f"🤖 {random.choice(responses)} (confidence {score:.2f})")
            end=time.time()
            print(end-startt)
        else:
            st.warning("🤖 I don’t know that. Can you teach me?")
            suggestions = bot.suggest(user_query)
            if suggestions:
                st.info(f"Did you mean: '{suggestions[0]}'?")

            new_response = st.text_input("Teach me the correct response:", key="teach_input")
            if new_response:
                bot.learn(user_query, new_response)
                st.success("🤖 Thanks! I learned something new.")

if __name__ == "__main__":
    
    main()
    
    
