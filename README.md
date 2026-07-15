# Assignment 3A — LangChain Internals: Marco, the City Trip Planner

## Overview

**Marco** is a travel-guide chatbot built to exercise three core LangChain building blocks
from the ground up: composed chains, conversational memory, and tool-using agents. It's built
around two pieces:

1. Two **chained** LLM calls that turn a destination + month + travel style + language into a
   personalised trip framing.
2. A **tool-using conversational loop** — Marco — that answers open-ended follow-up questions
   about flights, events, hotels, and visas, using four purpose-built tools and a persistent
   memory of the conversation.

---

## Part 1 — Chains: Prompt → LLM → Parser, Then a Second Chain Off the First

### Chain 1 — Travel Season Classifier

A tightly-constrained chain that takes a city and month and returns one formatted line —
season type, weather, crowd level, and a travel note — with a strict output format enforced
entirely through prompt instructions and few-shot examples:

```python
prompt1 = ChatPromptTemplate.from_messages([
    ("system", """
You are a travel season classifier. Your only job is to return a
single formatted line describing the travel season for a given city
and month.

Output format (follow exactly):
"<City> in <Month> — <season type>, <weather>, <crowd level>, <one travel note>"
...
Examples:
Rome in July — Peak season, hot and sunny, very busy, book early
Prague in November — Off-peak, cold and grey, quiet, budget-friendly
..."""),
    ("human", "City: {city}\nMonth: {month}")
])

chain1 = prompt1 | llm | StrOutputParser()
```

```python
chain1.invoke({"city": "Morocco", "month": "December"})
```

`StrOutputParser()` here strips the LangChain message wrapper (an `AIMessage` object with
metadata, token usage, etc.) down to just the plain string content — the one line of text
Chain 2 needs as input.

### Chain 2 — Trip Framing Advisor

Chain 2 takes Chain 1's output plus two fields that bypass Chain 1 entirely — `travel_style`
and `language` — and writes a short, personalised paragraph in the requested language:

```python
chain2 = (
    {
        "profile":      chain1,
        "travel_style": itemgetter("travel_style"),
        "language":     itemgetter("language"),
    }
    | prompt2
    | llm
    | StrOutputParser()
)
```

```python
# Test 1 — German output, budget traveller
chain2.invoke({
    "city": "Rome", "month": "July",
    "travel_style": "budget solo", "language": "German"
})

# Test 2 — Italian output, luxury couple
chain2.invoke({
    "city": "Barcelona", "month": "April",
    "travel_style": "luxury couple", "language": "Italian"
})
```

**Why Chain 1 lives inside the dictionary instead of being called first, separately:**
Writing `{"profile": chain1, ...}` makes Chain 1 a `Runnable` that LangChain executes as part
of the *same* composed pipeline, in parallel with the two `itemgetter` lookups. The whole
thing — Chain 1 running, `travel_style`/`language` passing through, and the results merging
into Chain 2's prompt — becomes a single object you can `.invoke()` once with the raw input
dict (`city`, `month`, `travel_style`, `language`). If Chain 1 were called manually first and
its output passed in as a plain string, you'd lose that: it becomes two disconnected
imperative steps instead of one traceable pipeline, and you'd have to re-wire the plumbing
by hand instead of LangChain resolving each key of the dict automatically.

---

## Part 2 — Memory: A Unified Runnable Memory

The assignment describes three classic, separately-configured memory objects
(`ConversationBufferMemory`, `ConversationSummaryMemory`, `VectorStoreMemory`) run side by
side against the same 20-turn conversation. What's implemented here instead is a single
memory mechanism built on LangChain's current recommended pattern —
`RunnableWithMessageHistory` — rather than the older memory classes:

```python
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_memory = RunnableWithMessageHistory(
    marco_runnable,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)
```

Every call is scoped to a `session_id`; the full, unsummarised message list for that session
is looked up, injected into the prompt as `{history}`, and appended to after each turn. This
is functionally closest to `ConversationBufferMemory` (full history, nothing dropped or
condensed) but wired through the modern `Runnable` interface, which is what current LangChain
docs recommend over the standalone memory classes going forward.

### Demonstrated across two live sessions

**Session `user_001` — Venice trip:**
```python
config = {"configurable": {"session_id": "user_001"}}

chain_with_memory.invoke({"question": "i would like to travel to venice in november"}, config=config)
chain_with_memory.invoke({"question": "how much are flights from germany to venice in November?"}, config=config)
chain_with_memory.invoke({"question": "what are good things to do there?"}, config=config)
chain_with_memory.invoke({"question": "suggest a good hotel to stay, give me names, ranges between 40 euro to 100 euro a night?"}, config=config)
chain_with_memory.invoke({"question": "the best restaurants to go to in venice or museum"}, config=config)
```

By the third turn ("what are good things to do **there**"), Marco resolves "there" to Venice
purely from history — the destination is never repeated by the user after turn 1.

**Session `user_002` — Japan trip (run independently, proving session isolation):**
```python
config = {"configurable": {"session_id": "user_002"}}

chain_with_memory.invoke({"question": "I want to visit japan, when is the best time to visit"}, config=config)
chain_with_memory.invoke({"question": "I want to visit museum, try out different dishes"}, config=config)
chain_with_memory.invoke({"question": "Kyoto sounds like a nice place"}, config=config)
```

`user_002`'s history never mixes with `user_001`'s Venice conversation — each `session_id`
gets its own `ChatMessageHistory` instance from the `store` dict.

### How this compares to the three classic memory types

| Memory type | What it keeps | Best for | Trade-off |
|---|---|---|---|
| **ConversationBufferMemory** | Every message, verbatim, forever | Short-to-medium conversations where nothing can be lost | Prompt grows every turn — eventually blows the context window and costs more per call |
| **ConversationSummaryMemory** | A running LLM-generated summary instead of raw turns | Long conversations where the gist matters more than exact wording | Costs an extra LLM call per turn to re-summarise; can lose specific numbers/names (e.g. exact flight prices) in the compression |
| **VectorStoreMemory** | Embeddings of past turns, retrieved by similarity to the current question | Very long-running or multi-session history where only relevant *fragments* matter | Retrieval can miss a relevant fact if it's phrased differently than the query; adds embedding/vector-store infrastructure |
| **Runnable memory (implemented)** | Every message, verbatim, per session | Same strength as Buffer memory — exact facts (prices, dates, hotel names) are never lost within a session | Same growth problem as Buffer memory over very long sessions; no built-in summarisation or similarity search |

For a trip planner specifically — where a wrong flight price or hotel name quoted back to the
user is worse than a slightly long prompt — keeping full, unsummarised history (Buffer-style,
which is what's implemented) is the safer default for the length of conversation shown here.
Summary or vector memory would start to earn their keep once conversations regularly ran past
Buffer memory's practical limit (dozens of turns across many sessions), where they'd trade
some precision for a prompt that doesn't keep growing.

---

## Part 3 — Tools: Four Tavily-Backed Tools

The assignment's suggested pairing — a web-search tool and a calculator — contrasts a
"look something up" tool against a "compute something" tool. What's implemented here takes a
related but different angle: **one underlying capability (Tavily web search) wrapped four
separate times**, each with a distinct name, description, and structured Pydantic output
schema, to test whether tool *selection* is really driven by the tool's description rather
than by what it does under the hood:

```python
tools = [flight_price_checker, local_events_finder, hotel_finder, visa_requirement_checker]
```

| Tool | Triggers on | Output schema |
|---|---|---|
| `flight_price_checker` | flight cost, airlines, "how do I get there" | `FlightAgentResponse` → list of `FlightResult` (airline, date, price) + sources |
| `local_events_finder` | things to do, festivals, food, sightseeing | `EventAgentResponse` → list of `EventResult` (name, date, venue, entry cost) + sources |
| `hotel_finder` | where to stay, accommodation cost | `HotelAgentResponse` → list of `HotelResult` (name, area, price, highlights) + sources |
| `visa_requirement_checker` | visas, entry requirements | `VisaAgentResponse` → requirement string (Visa-free / eVisa available / Visa required / Check official source) + sources |

No calculator is included — all four tools are "look-up" tools distinguished purely by
intent, rather than a look-up tool versus a compute tool. That means the demo shows
description-driven selection *among lookups*, rather than the sharper look-up-vs-calculation
contrast the assignment describes.

### The agent deciding which tool to use

Marco runs a manual tool loop rather than a prebuilt LangChain agent, so the decision process
is fully visible:

```python
def marco_tool_loop(inputs: dict) -> str:
    question = inputs["question"]
    history  = inputs.get("history", [])

    messages = [SystemMessage(content=MARCO_SYSTEM)] + history + [HumanMessage(content=question)]

    for iteration in range(1, MAX_ITERATIONS + 1):
        ai_message = llm_with_tools.invoke(messages)
        messages.append(ai_message)

        if not ai_message.tool_calls:
            return ai_message.content

        for tool_call in ai_message.tool_calls:
            selected    = tools_dict[tool_call["name"]]
            tool_result = selected.invoke(tool_call["args"])
            messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))

    return "Kindly reframe your question."
```

Running through the Venice session turn by turn, the tool the model reaches for tracks the
intent of each question exactly as the system prompt describes:

- *"how much are flights from germany to venice in November?"* → `flight_price_checker`
- *"what are good things to do there?"* → `local_events_finder`
- *"suggest a good hotel to stay..."* → `hotel_finder`
- *"the best restaurants to go to in venice or museum"* → `local_events_finder` again (restaurants/museums fall under "things to do")

This mirrors the assignment's underlying goal — showing the agent picking the right tool for
the right intent — just spread across four candidates sharing one search backend instead of
two tools with genuinely different mechanisms.

---

## Tech stack

| Piece | Tool |
|---|---|
| LLM | `gpt-4o-mini` via `langchain_openai.ChatOpenAI` (`temperature=0`) |
| Search backend | Tavily (`tavily.TavilyClient`), wrapped by all four tools |
| Structured tool output | Pydantic `BaseModel` schemas per tool |
| Chaining | LangChain LCEL (`|` pipe syntax), `itemgetter` for selective field pass-through |
| Memory | `RunnableWithMessageHistory` + `ChatMessageHistory`, keyed by `session_id` |
| UI | Streamlit (`app.py`) — name-gated session start, styled chat bubbles, reasoning happens server-side |
| Secrets | `python-dotenv` (`.env`: `OPENAI_API_KEY`, `TAVILY_API_KEY`) |

## Setup

```bash
pip install langchain langchain-openai langchain-tavily tavily-python \
            streamlit pydantic python-dotenv
```

`.env`:
```
OPENAI_API_KEY=your-key-here
TAVILY_API_KEY=your-key-here
```

Run the app:
```bash
streamlit run app.py
```
