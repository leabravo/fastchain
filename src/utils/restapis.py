import requests
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from langchain.schema import Document
from langchain.text_splitter import TokenTextSplitter

class StockDetails(BaseModel):
    tipoDoc: int
    nroDoc: str
    sexo: str
    prefijoCompania: str
    estado: int
    antiguedadNoVigentes: Optional[int] = None


sample_body = StockDetails(
    tipoDoc = 1,
    nroDoc = "23183381",
    sexo = "M",
    prefijoCompania = "BG",
    estado = 1,
    #antiguedadNoVigentes = None
)

# Decorar con fn augment_with_other_api que itere sobre los resultados
# de stock details y agregue los datos de cláusulas y anexos
def post_to_rest_api(url: str, model: BaseModel) -> List[Dict[str, Any]]:
    response = requests.post(url, headers={'Content-Type': "application/json"}, data=model.json())
    print(response.text)
    return response.json()

def split_text_by_tokens(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def make_documents_from_dict(response: List[Dict[str, Any]]) -> List[Document]:
    doc_list = []

    for item in response['result']:
        placeholder_clausulas = "Los productos del usuario son Auto, Celular y Compra Protegida."
        for split in split_text_by_tokens(placeholder_clausulas, 100, 2):
            doc_list.append(
                Document(
                    page_content=split,
                    metadata={k: v for k, v in item.items() if type(v) in [str, int, float]}
                )
            )

    return doc_list
