"""Time travel: rileggere la storia passata e ripartire da li."""


from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from agent_a_mano import builder

agent=builder.compile(checkpointer=InMemorySaver())
THREAD={"configurable":{"thread_id":"ordine-7"}}

if __name__ == "__main__":
    print("--- 1. Semplice conversazione")
    r = agent.invoke(
        {"messages": [HumanMessage("2 laptop scontati del 10%: totale?")]},
        config=THREAD
    )
    print(" ", r["messages"][-1].content)

    print("--- 2. Tutti i checkpoint salvati")
    storia = list(reversed(list(agent.get_state_history(THREAD))))
    for i, s in enumerate(storia):
        print(f"  [{i}] messaggi={len(s.values['messages']):2}  prossimo nodo: {s.next or ('END',)}")

    print("--- 3. Rileggiamo uno stato passato (il 2)")
    passato = storia[2]
    for m in passato.values["messages"]:
        tc = getattr(m, "tool_calls", None)
        print(f"  {type(m).__name__:12}", [c["name"] for c in tc] if tc else m.content[:60])

    print("--- 4. Cambiamo domanda e ripartiamo da lì")
    nuovo = agent.update_state(
        passato.config,
        {"messages": [HumanMessage("Anzi 5 laptop, senza sconto")]},
        as_node="__start__" #obbligatorio
    )
    r2 = agent.invoke(None,config=nuovo) #None = continua, non aggiungere input
    print(" ", r2["messages"][-1].content)

    print("--- 5. Rivediamo la storia")
    print(f"  checkpoint totali ora: {len(list(agent.get_state_history(THREAD)))}")
    storia = list(reversed(list(agent.get_state_history(THREAD))))
    for i, s in enumerate(storia):
        print(f"  [{i}] messaggi={len(s.values['messages']):2}  prossimo nodo: {s.next or ('END',)}")