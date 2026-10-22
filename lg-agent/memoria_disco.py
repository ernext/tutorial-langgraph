"""Memorizziamo su file in modo da sopravvivere al riavvio"""

from pathlib import Path
import sqlite3

from langchain.messages import HumanMessage
from agent_a_mano import builder
from langgraph.checkpoint.sqlite import SqliteSaver


DB = Path(__file__).parent / "memoria.db"
THREAD = {"configurable": {"thread_id": "cliente-42"}}

if __name__ == "__main__":
    primo_avvio = not DB.exists()

    with sqlite3.connect(DB, check_same_thread=False) as conn:
        agent = builder.compile(checkpointer=SqliteSaver(conn))

        if primo_avvio:
            print("Primo avvio: No DB")
            domanda = "Quanto costano 2 laptop e 3 monitor?"
        else:
            print("Secondo avvio: riprendo dal DB")
            domanda = "Quanti monitor ti avevo chiesto? Aggiungi un monitor e dammi il vecchi e il nuovo totale?"

        print(f"  Q: {domanda}")
        risposta = agent.invoke({"messages": [HumanMessage(domanda)]}, config=THREAD)
        print(f"  A: {risposta['messages'][-1].content}")

        stato = agent.get_state(THREAD)
        print(f"\nmessaggi nel thread: {len(stato.values['messages'])}")

        if primo_avvio:
            print("\n>> Rilancia lo stesso comando e guarda cosa succede.")