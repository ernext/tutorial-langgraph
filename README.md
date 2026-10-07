# Tutorial LangGraph

Codice del tutorial: costruire agent AI con **LangGraph**, passo per passo,
usando un **modello locale** (nessuna API key, nessun costo).

Stack: Python 3.12 · [uv](https://docs.astral.sh/uv/) · LangGraph 1.2 ·
LangChain 1.4 · [Ollama](https://ollama.com)

## Prerequisiti

1. **Ollama** con un modello che sappia chiamare i tool. Nei video uso
   `gemma4:12b-mlx`; vanno bene anche `qwen3:8b` o `llama3.1:8b`.

   ```bash
   ollama list
   ```

2. **uv**:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

## Avvio

```bash
git clone https://github.com/ernext/tutorial-langgraph.git
cd tutorial-langgraph/lg-agent
uv sync
```

`uv sync` scarica Python 3.12 e installa le dipendenze esatte dal lockfile:
non tocca il Python di sistema.

Verifica che il tuo modello sappia chiamare i tool — senza, il resto non funziona:

```bash
uv run python verifca.py
```

## I file

| file | lezione | cosa mostra |
|---|---|---|
| `verifca.py` | 00 | il modello locale risponde e sa chiamare i tool |
| `agent_a_mano.py` | 01-02 | un agent scritto a mano: stato, nodi, archi, ciclo ReAct |
| `create-agent.py` | 03 | lo stesso agent con `create_agent`: 40 righe diventano 3 |

Ogni file è eseguibile da solo:

```bash
uv run python agent_a_mano.py
```

> Il repo cresce lezione per lezione: ogni commit corrisponde a un video.

## L'idea di fondo

> LangGraph non è una libreria di AI: è una **macchina a stati**. Tu definisci dei
> nodi (funzioni che aggiornano uno stato condiviso) e degli archi (chi viene dopo
> chi). Il modello è solo uno dei nodi possibili.

E il corollario che scoprirai nella lezione 02:

> Quando un agent si comporta male, nove volte su dieci la colpa non è del grafo
> né del prompt: è di come hai progettato i tool.

## Se cambi modello

Una riga per file:

```python
llm = ChatOllama(model="gemma4:12b-mlx", temperature=0)
```

Con un modello più piccolo il tool calling è meno affidabile: se un esempio non
parte, prima verifica con `verifca.py`.

## Licenza

[MIT](LICENSE) — fai quello che vuoi, è materiale didattico.
