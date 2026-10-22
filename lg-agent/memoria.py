"""L'agent mantiene la memoria della conversazione: checkpointer e thread_id."""

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from agent_a_mano import builder #stesso grafo della lezione precedente

senza_memoria = builder.compile()
con_memoria = builder.compile(checkpointer=InMemorySaver()) #memorizza stato completo del grafo. (tutto in RAM)

def chiedi(agent, testo: str, thread:str | None = None) -> None: 
    config ={"configurable":{"thread_id": thread}} if thread else None
    risposta = agent.invoke({"messages": [HumanMessage(testo)]}, config=config)
    print(f" Q: {testo}")
    print(f" A: {risposta['messages'][-1].content}\n")

if __name__ == "__main__":
    # print("--- SENZA checkpointer ---")
    # chiedi(senza_memoria, "Quanto costano due laptop?")
    # chiedi(senza_memoria, "Quanti ne avevo chiesti?")

    print("--- CON checkpointer, thread cliente-42 ---")
    chiedi(con_memoria, "Quanto costano due laptop?", "cliente-42")
    chiedi(con_memoria, "Quanti ne avevo chiesti?", "cliente-42")

    # print("--- CON checkpointer, thread cliente-99 ---")
    # chiedi(con_memoria, "Quanti ne avevo chiesti?", "cliente-99")

    print("--- cosa c'e' dentro cliente-42 ---")
    stato = con_memoria.get_state({"configurable": {"thread_id": "cliente-42"}})
    for m in stato.values["messages"]:
        tc = getattr(m, "tool_calls", None)
        print(f"  {type(m).__name__:12}", [c["name"] for c in tc] if tc else m.content[:60])