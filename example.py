import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI
from transformers import pipeline
from dotenv import load_dotenv
import os
load_dotenv()

st.title("Chatbot with text-classification")

if "chat_started" not in st.session_state:
    st.session_state["chat_started"] = True
    st.session_state["chat_history"] = []
    st.session_state['chat_anaylzer'] = pipeline('text-classification', model="monologg/koelectra-base-finetuned-nsmc")


llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 훌륭한 어시스턴트야"),
    MessagesPlaceholder("history"),
    ("user", "{query}")
]
)

if st.session_state["chat_started"]:
    chain = prompt | llm
    if query := st.chat_input():
        result = chain.invoke({
            "history": st.session_state["chat_history"],
            "query": query})

        for i in range(len(st.session_state["chat_history"])):
            if i % 2 == 0:
                with st.chat_message("user"):
                    st.markdown(st.session_state["chat_history"][i])
            else:
                with st.chat_message("assistant"):
                    st.markdown(st.session_state["chat_history"][i])

        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            st.markdown(result.content)
            st.markdown("This message is: " + st.session_state["chat_anaylzer"](result.content)[0]['label'])

        st.session_state["chat_history"].append(query)
        st.session_state["chat_history"].append(result.content)
