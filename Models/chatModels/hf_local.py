from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

os.environ['HF_HOME']='D:/Installed Softwares/Gen AI/chatModels/cache'
llm=HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 150}
)

model=ChatHuggingFace(llm=llm)

result=model.invoke("What is the capital of Pakistan?")
print(result.content)