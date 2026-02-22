from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate  # Import this
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.header('Research Tool')

# 1. Initialize the Model
# Make sure OPENAI_API_KEY is in your .env file
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Taking input from user
paper_input = st.selectbox("Select research paper name", ["Paper A", "Paper B", "Paper C"])
style_input = st.selectbox("Select summary style", ["Concise", "Detailed", "Bullet Points"])

# 2. Define the Prompt Template correctly
template = PromptTemplate(
    template="You are a helpful research assistant. Summarize the {paper_input} research paper in a {style_input} style.",
    input_variables=["paper_input", "style_input"]
)

if st.button('Summarize'):
    # 3. Invoke the chain (Prompt -> Model)
    # We format the prompt first, then pass it to the model
    formatted_prompt = template.invoke({'paper_input': paper_input, 'style_input': style_input})
    
    result = model.invoke(formatted_prompt)
    
    # 4. Display result on the webpage, not the terminal
    st.write(result.content)