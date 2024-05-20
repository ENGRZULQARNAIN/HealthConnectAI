import json
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


class Document(BaseModel):
    page_content:str
    metadata:Dict

def load_documents(path="data.json"):
    with open(path, "r") as f:
        data = json.load(f)


    docs=[]
    for index,doc in enumerate(data):
        docs.append(Document(page_content=str(doc),metadata={'source': '/content/data.json', 'seq_num': index+1}))

    return docs


def create_embedings(persist_directory="",docs=load_documents()):
    # save to disk
    db = Chroma.from_documents(docs, embeddings, persist_directory="docters_db")
    return  db.as_retriever(search_type="similarity", search_kwargs={"k": 5})


