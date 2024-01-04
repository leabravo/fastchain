from api.templates.dataclasses import Templates
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.prompts import PromptTemplate
from langchain.utils.math import cosine_similarity

def create_template(templates: Templates) -> ChatPromptTemplate:
    all_templates = []
    for template in templates.templates:
        if template.template_type == "human":
            all_templates.append(HumanMessagePromptTemplate.from_template(template.template))
        elif template.template_type == "system":
            all_templates.append(SystemMessagePromptTemplate.from_template(template.template))
        else:
            all_templates.append(AIMessagePromptTemplate.from_template(template.template))
        
    chat_prompt = ChatPromptTemplate.from_messages(all_templates)
    return chat_prompt

def popul8_template():
    return True
