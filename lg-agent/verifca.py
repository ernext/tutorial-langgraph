"""Il modello locale risponde e sa chiamare i tool???"""

from langchain_core.tools import tool
from langchain_ollama import ChatOllama

MODELLO="gemma4:12b-mlx"

@tool
def somma(a: int, b:int) -> int:
    """Somma due numeri interi"""
    return a+b


llm = ChatOllama(model=MODELLO, temperature=0)

print("1. chat ->", llm.invoke("Rispondi solo: OK").content)

risposta = llm.bind_tools([somma]).invoke("Quanto fa 18 + 23? Usa il tool.")

if risposta.tool_calls:
    print("2. tools -> OK:", risposta.tool_calls)
else:
    print("2. tools -> NON supportato. Ha risposto: ", risposta.content)