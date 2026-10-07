"""Stesso agent ma usando create_agent: 40 righe precedenti diventano 3"""

from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from agent_a_mano import SYSTEM, TOOLS, mostra


agent = create_agent(
    model=ChatOllama(model="gemma4:12b-mlx", temperature=0),
    tools=TOOLS,
    system_prompt=SYSTEM.content
)

if __name__ == "__main__":
    print("--- create_agent ---")
    print(agent.get_graph().draw_ascii())
    mostra(agent, "Due laptop, tre monitor, scontati del 22%: totlae?")
    mostra(agent, "Quanti laptop ho chiesto?")
