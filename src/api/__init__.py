from fastapi import APIRouter, Depends
from langchain.schema import AIMessage, HumanMessage
from api.models  import router as models_router
from api.context import router as context_router
from api.templates import router as prompts_router
from src.api.dataclasses import ChatRequest, VariableClassifier
from src.db.orm.models import Template
from src.db.orm import get_db
from src.utils.restapis import sample_body, post_to_rest_api, make_documents_from_dict

from sqlalchemy import select
from sqlalchemy.orm import Session
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate
)
from langchain.chat_models import AzureChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.embeddings import OpenAIEmbeddings
from langchain.utils.math import cosine_similarity
from langchain.vectorstores import Chroma

import os

path = os.path.basename(os.path.dirname(os.path.realpath(__file__)))

router = APIRouter(
    prefix=f"/{path}"
)

router.include_router(models_router)
router.include_router(prompts_router)
router.include_router(context_router)

model = AzureChatOpenAI(temperature=0, deployment_name="chat")
embeddings = OpenAIEmbeddings()

@router.get("/")
async def index():
    return { "path": path }

@router.post("/chat")
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    # Si el request tiene más de un template, entonces usar condicional, sino
    # usar el prompt directamente
    sys_messages = []
    chat_prompts = []

    for template_data in request.prompts:
        stmt = select(Template).where(Template.id == template_data.template_id)
        template: Template = db.scalar(stmt)
        sys_prompt = template.system_template.format(**template_data.system_variables)
        chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(sys_prompt),
            SystemMessagePromptTemplate.from_template("\nAnswer with the following context: {context}\n"),
            HumanMessagePromptTemplate.from_template(template.human_template),
            #AIMessagePromptTemplate.from_template(template.ai_template) el AIMessage lo usaré para los examples
        ])

        sys_messages.append(sys_prompt)
        #messages = chat_prompt.format_prompt(**template_data.system_variables,
        #                                     input = request.question).to_string()
        chat_prompts.append(chat_prompt)

    print(chat_prompts)
        
    prompt_embeddings = embeddings.embed_documents(sys_messages)
        
    def prompt_router(input):
        query_embedding = embeddings.embed_query(input["input"])
        similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
        most_similar = chat_prompts[similarity.argmax()]
        print(f"ROUTED: {most_similar}")
        return most_similar
        
    
    res = post_to_rest_api(
        "https://svc.galiciaseguros.com.ar/insurance-holding-service/v2/stockDetails",
        sample_body)

    docs = make_documents_from_dict(res)
    print("Made DOCS")
    
    vecdb = Chroma.from_documents(docs, OpenAIEmbeddings())
    retriever = vecdb.as_retriever()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    print("Chroma OK!")
    print(retriever.invoke(request.question))
    
    chain = (
        {'context': retriever | format_docs,'input': RunnablePassthrough()}
        | RunnableLambda(prompt_router)
        | model
        | StrOutputParser()
    )

    return chain.invoke(request.question)
