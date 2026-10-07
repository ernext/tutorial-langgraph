"""Un primo agent scritto a mano: 2 nodi, un arco condizionale, un ciclo"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph

CATALOGO = {"laptop": 1200.0,"mouse": 100.0,"monitor": 500.0,"tastiera": 200.0,}

@tool
def cerca_prezzo(prodotto: Literal["laptop", "monitor", "tastiera", "mouse"]) -> float:
    """Prezzo di listino unitario di un articolo del catalogo, in euro."""
    return CATALOGO[prodotto]

@tool
def applica_sconto_e_totale(prezzo_unitario: float,quantita: int = 1, percentuale: float = 0) -> float:
    """Totale di una riga d'ordine: prezzo unitario x quantita, meno lo sconto in percentuale"""
    return round(prezzo_unitario * quantita *(1 - percentuale / 100), 2)

TOOLS=[cerca_prezzo, applica_sconto_e_totale]
TOOLS_BY_NAME={t.name: t for t in TOOLS}

SYSTEM = SystemMessage("Sei l'assistente commerciale di un negozio di informatica. Non fare calcoli a mente ma usa sempre i tool.")

llm = ChatOllama(model="gemma4:12b-mlx", temperature=0).bind_tools(TOOLS)

#-----NODI

def nodo_modello(state: MessagesState) -> dict:
    return {"messages": [llm.invoke([SYSTEM] + state["messages"])]}

def nodo_tools(state: MessagesState) -> dict:
    risultati = []
    for chiamata in state["messages"][-1].tool_calls:
        output = TOOLS_BY_NAME[chiamata["name"]].invoke(chiamata["args"])
        risultati.append(ToolMessage(content=str(output), tool_call_id=chiamata["id"]))
    return {"messages": risultati}

#------ ARCO CONDIZIONALE

def deve_usare_tool(state: MessagesState) -> str:
    return "tools" if state["messages"][-1].tool_calls else "fine"

#------------ COSTRUZIONE DEL GRAFO

builder = StateGraph(MessagesState)
builder.add_node("modello", nodo_modello)
builder.add_node("tools", nodo_tools)

builder.add_edge(START, "modello")
builder.add_conditional_edges("modello", deve_usare_tool,{"tools":"tools", "fine": END})
builder.add_edge("tools","modello")

agent = builder.compile()

#----------- DEMO

def mostra(agent, domanda: str) -> None:
    print(f"\n> {domanda}")
    for passo in agent.stream({"messages": [HumanMessage(domanda)]}, stream_mode="updates"):
        for nodo, aggiornamento in passo.items():
            for m in aggiornamento["messages"]:
                tc = getattr(m, "tool_calls", None)
                corpo = [(c["name"], c["args"]) for c in tc] if tc else m.content
                print(f"  {nodo:8} {corpo}")


if __name__ == "__main__":
    print(agent.get_graph().draw_ascii())
    # mostra(agent, "Il laptop a catalogo, scontato del 15%: quanto viene?")
    # mostra(agent, "Il laptop, scontato del 15%: quanto viene?")
    # mostra(agent, "Quanto costano due laptop scontato del 12%?")
    mostra(agent, "Quanto costano due laptop 2 mouse e 3 tastiere e 1 monitor tutto scontato del 12%?")
    mostra(agent, "Quanto costano due speacker scontato del 12%?")



