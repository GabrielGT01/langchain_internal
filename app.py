

import os
import re
import streamlit as st
from tavily import TavilyClient
from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Marco — Your Travel Guide",
    page_icon="🌍",
    layout="centered",
)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0f0e0c;
    color: #f0ebe0;
}

/* ── Hide default streamlit elements ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 780px; }

/* ── Hero header ── */
.marco-hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: linear-gradient(135deg, #1a1508 0%, #0f0e0c 60%, #1a1508 100%);
    border-bottom: 1px solid #3a2e1a;
    margin-bottom: 1.5rem;
}
.marco-hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 700;
    color: #c9a84c;
    letter-spacing: -0.5px;
    margin: 0;
    line-height: 1.1;
}
.marco-hero p {
    font-size: 1rem;
    color: #a09070;
    margin: 0.5rem 0 0;
    font-weight: 300;
    letter-spacing: 0.5px;
}

/* ── Welcome card ── */
.welcome-card {
    background: linear-gradient(135deg, #1e1a0f, #151208);
    border: 1px solid #3a2e1a;
    border-left: 4px solid #c9a84c;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.5rem;
    font-size: 0.95rem;
    color: #d4c9a8;
    line-height: 1.6;
}
.welcome-card strong { color: #c9a84c; }

/* ── Name input section ── */
.name-section {
    background: #151208;
    border: 1px solid #2a2010;
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    text-align: center;
}
.name-section h3 {
    font-family: 'Playfair Display', serif;
    color: #c9a84c;
    margin-bottom: 0.5rem;
    font-size: 1.3rem;
}
.name-section p {
    color: #806a40;
    font-size: 0.85rem;
    margin-bottom: 1rem;
}

/* ── Streamlit input override ── */
.stTextInput input {
    background-color: #1a1508 !important;
    border: 1px solid #3a2e1a !important;
    border-radius: 6px !important;
    color: #f0ebe0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1rem !important;
}
.stTextInput input:focus {
    border-color: #c9a84c !important;
    box-shadow: 0 0 0 2px rgba(201, 168, 76, 0.15) !important;
}
.stTextInput label { color: #a09070 !important; font-size: 0.85rem !important; }

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #c9a84c, #a07830) !important;
    color: #0f0e0c !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 0.5rem 1.5rem !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Chat messages ── */
.stChatMessage {
    background: transparent !important;
    border: none !important;
}

/* ── User message bubble ── */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: #1a1508 !important;
    border: 1px solid #2a2010 !important;
    border-radius: 12px !important;
    padding: 0.8rem 1rem !important;
    margin-bottom: 0.5rem !important;
}

/* ── Marco message bubble ── */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: #151208 !important;
    border: 1px solid #3a2e1a !important;
    border-left: 3px solid #c9a84c !important;
    border-radius: 12px !important;
    padding: 0.8rem 1rem !important;
    margin-bottom: 0.5rem !important;
}

/* ── Chat input ── */
.stChatInputContainer {
    background: #151208 !important;
    border: 1px solid #3a2e1a !important;
    border-radius: 10px !important;
}
.stChatInputContainer textarea {
    background: transparent !important;
    color: #f0ebe0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Divider ── */
hr { border-color: #2a2010 !important; margin: 1rem 0 !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #c9a84c !important; }

/* ── Session badge ── */
.session-badge {
    display: inline-block;
    background: #1a1508;
    border: 1px solid #3a2e1a;
    border-radius: 20px;
    padding: 0.3rem 0.8rem;
    font-size: 0.78rem;
    color: #c9a84c;
    margin-bottom: 1rem;
}

/* ── Clear button ── */
.clear-btn > button {
    background: transparent !important;
    border: 1px solid #3a2e1a !important;
    color: #806a40 !important;
    font-size: 0.8rem !important;
    padding: 0.3rem 0.8rem !important;
}
.clear-btn > button:hover {
    border-color: #c9a84c !important;
    color: #c9a84c !important;
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CLIENTS & SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def init_clients():
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return tavily, llm

tavily, llm = init_clients()

class Source(BaseModel):
    url: str = Field(description="The URL of the source")

class FlightResult(BaseModel):
    flight_name: str = Field(description="Airline and route")
    flight_date: str = Field(description="Date of the flight")
    price:       str = Field(description="Price as a string")

class FlightAgentResponse(BaseModel):
    answer:  str                = Field(description="Plain-English summary")
    flights: List[FlightResult] = Field(default_factory=list)
    sources: List[Source]       = Field(default_factory=list)

class EventResult(BaseModel):
    event_name: str = Field(description="Full name of the event")
    event_date: str = Field(description="Date or date range")
    venue:      str = Field(description="Venue or area")
    entry_cost: str = Field(description="Entry cost")

class EventAgentResponse(BaseModel):
    answer:  str               = Field(description="Plain-English summary")
    events:  List[EventResult] = Field(default_factory=list)
    sources: List[Source]      = Field(default_factory=list)

class HotelResult(BaseModel):
    hotel_name: str = Field(description="Name of the hotel")
    area:       str = Field(description="Neighbourhood or area")
    price:      str = Field(description="Price per night")
    highlights: str = Field(description="What makes it stand out")

class HotelAgentResponse(BaseModel):
    answer:  str               = Field(description="Plain-English summary")
    hotels:  List[HotelResult] = Field(default_factory=list)
    sources: List[Source]      = Field(default_factory=list)

class VisaAgentResponse(BaseModel):
    answer:      str          = Field(description="Clear visa answer")
    requirement: str          = Field(description="Visa status")
    sources:     List[Source] = Field(default_factory=list)

# ══════════════════════════════════════════════════════════════════════════════
# TOOLS
# ══════════════════════════════════════════════════════════════════════════════
@tool
def flight_price_checker(query: str) -> dict:
    """
    Find flight prices between two cities for a specific month the user selected.
    Input must include: origin city, destination city, and month.
    Use for any question about flight cost, airlines, or reaching a destination.
    """
    try:
        raw = tavily.search(query=query, max_results=5)
    except Exception as e:
        return FlightAgentResponse(answer="Flight search unavailable.", flights=[], sources=[]).model_dump()

    flights, sources = [], []
    for r in raw.get("results", []):
        content  = r.get("content", "")
        prices   = re.findall(r'[\$€£]\d+(?:\.\d{1,2})?', content)
        dates    = re.findall(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?', content)
        airlines = re.findall(r'\b(Ryanair|EasyJet|Lufthansa|Wizz Air|Vueling|Air France|KLM|Iberia|Turkish Airlines|British Airways)\b', content)
        if prices:
            flights.append(FlightResult(
                flight_name=airlines[0] + " flight" if airlines else r.get("title", "Unknown airline"),
                flight_date=dates[0] if dates else "Check site for dates",
                price=prices[0],
            ))
        sources.append(Source(url=r.get("url", "")))

    return FlightAgentResponse(answer=f"Flights found for: {query}", flights=flights, sources=sources).model_dump()


@tool
def local_events_finder(query: str) -> dict:
    """
    Find local events, festivals, restaurants, museums and activities
    in a city for the month the user selected.
    Input must include: city, month, and event type.
    Use for any question about things to do or local experiences.
    """
    try:
        raw = tavily.search(query=query, max_results=5)
    except Exception as e:
        return EventAgentResponse(answer="Events search unavailable.", events=[], sources=[]).model_dump()

    events, sources = [], []
    for r in raw.get("results", []):
        content = r.get("content", "")
        costs   = re.findall(r'(?:free|Free|FREE|[\$€£]\d+(?:\.\d{1,2})?)', content)
        events.append(EventResult(
            event_name=r.get("title", "Unknown event"),
            event_date=query,
            venue=r.get("url", "See source"),
            entry_cost=costs[0] if costs else "See link",
        ))
        sources.append(Source(url=r.get("url", "")))

    return EventAgentResponse(answer=f"Events found for: {query}", events=events, sources=sources).model_dump()


@tool
def hotel_finder(query: str) -> dict:
    """
    Find hotels in a city for the month the user selected.
    Input must include: city, month, and budget level (budget/mid-range/luxury).
    Use for any question about where to stay or accommodation cost.
    """
    try:
        raw = tavily.search(query=query, max_results=5)
    except Exception as e:
        return HotelAgentResponse(answer="Hotel search unavailable.", hotels=[], sources=[]).model_dump()

    hotels, sources = [], []
    for r in raw.get("results", []):
        content = r.get("content", "")
        prices  = re.findall(r'[\$€£]\d+(?:\.\d{1,2})?(?:\s?(?:per night|/night))?', content)
        hotels.append(HotelResult(
            hotel_name=r.get("title", "Unknown hotel"),
            area="See link",
            price=prices[0] if prices else "See link",
            highlights=content[:120] + "..." if len(content) > 120 else content,
        ))
        sources.append(Source(url=r.get("url", "")))

    return HotelAgentResponse(answer=f"Hotels found for: {query}", hotels=hotels, sources=sources).model_dump()


@tool
def visa_requirement_checker(query: str) -> dict:
    """
    Check visa requirements for travelling from one country to another.
    Input must include: user's nationality and destination country.
    Use for any question about visas or entry requirements.
    """
    try:
        raw = tavily.search(query=query, max_results=3)
    except Exception as e:
        return VisaAgentResponse(answer="Visa search unavailable.", requirement="Check official source", sources=[]).model_dump()

    full_content, sources = "", []
    for r in raw.get("results", []):
        full_content += r.get("content", "") + " "
        sources.append(Source(url=r.get("url", "")))

    content_lower = full_content.lower()
    if "visa-free" in content_lower or "no visa required" in content_lower:
        requirement = "Visa-free"
    elif "evisa" in content_lower or "e-visa" in content_lower:
        requirement = "eVisa available"
    elif "visa required" in content_lower or "must apply" in content_lower:
        requirement = "Visa required"
    else:
        requirement = "Check official source"

    return VisaAgentResponse(answer=f"Visa info for: {query}", requirement=requirement, sources=sources).model_dump()


tools      = [flight_price_checker, local_events_finder, hotel_finder, visa_requirement_checker]
tools_dict = {t.name: t for t in tools}

# ══════════════════════════════════════════════════════════════════════════════
# MARCO SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════
MARCO_SYSTEM = """
You are Marco — a well-travelled, warm, and witty travel guide who has personally
visited hundreds of cities. You speak like a friend who knows everything about a
destination: specific, honest, occasionally funny, never generic.

You have access to tools. Use them smartly:
- flight_price_checker     → user asks about flights or travel cost
- local_events_finder      → user asks about things to do, eat, or see
- hotel_finder             → user asks about where to stay
- visa_requirement_checker → user asks about visas or entry requirements

Tool rules:
- NEVER guess prices, hotel names, or availability. Always use the tool.
- Call the tool first, then weave the result into natural conversation.
- If you need more info before searching, ask first.
- You can call multiple tools in one turn if needed.

If a question is completely unrelated to travel, destinations, or trip planning,
respond with exactly: "Please ask Marco a travel-related question — I'm your guide, not a search engine!"

How to sound like Marco:
- Open with something vivid — a smell, a street, a crowd, a moment.
  Never "Great choice!" or "Absolutely!" — banned.
- Weave tool results into flowing prose. Never dump raw data at the user.
- Cover traveller angles naturally: solo, couple, family, group, budget, luxury.
- Be honest. If July is brutal, say so. If something is overrated, warn them.
- 4–8 sentences of flowing prose. No bullets. No headers. No lists.
- Close with one golden tip or one question to help them plan better.

Language rule: Always reply in the same language the user wrote in.
Your tone: a well-travelled friend at a café who just got back from this exact trip.
"""

# ══════════════════════════════════════════════════════════════════════════════
# TOOL LOOP + MEMORY
# ══════════════════════════════════════════════════════════════════════════════
llm_with_tools = llm.bind_tools(tools)
MAX_ITERATIONS = 10

def marco_tool_loop(inputs: dict) -> str:
    question = inputs["question"]
    history  = inputs.get("history", [])

    messages = (
        [SystemMessage(content=MARCO_SYSTEM)]
        + history
        + [HumanMessage(content=question)]
    )

    for _ in range(1, MAX_ITERATIONS + 1):
        ai_message = llm_with_tools.invoke(messages)
        messages.append(ai_message)

        if not ai_message.tool_calls:
            return ai_message.content

        for tool_call in ai_message.tool_calls:
            selected    = tools_dict[tool_call["name"]]
            tool_result = selected.invoke(tool_call["args"])
            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"]
                )
            )

    return "Please ask Marco a travel-related question for a better response!"


marco_runnable = RunnableLambda(marco_tool_loop)
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

@st.cache_resource
def build_chain():
    return RunnableWithMessageHistory(
        marco_runnable,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

chain_with_memory = build_chain()

def ask_marco(question: str, user_name: str) -> str:
    config = {"configurable": {"session_id": user_name.lower().strip()}}
    try:
        response = chain_with_memory.invoke({"question": question}, config=config)
        if not response or len(response.strip()) < 10:
            return "Please ask Marco a travel-related question — I'm your guide, not a search engine!"
        return response
    except Exception as e:
        return f"Marco hit a snag — please try rephrasing your question. ({str(e)[:80]})"

# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="marco-hero">
    <h1>🌍 Marco</h1>
    <p>Your personal travel guide — cities, flights, hotels & more</p>
</div>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "greeted" not in st.session_state:
    st.session_state.greeted = False

# ══════════════════════════════════════════════════════════════════════════════
# NAME GATE — shown until user enters name
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.user_name:
    st.markdown("""
    <div class="welcome-card">
        <strong>Welcome to Marco — your travel guide with expertise.</strong><br><br>
        Marco has personally visited hundreds of cities and speaks like a friend who
        just got back from your destination. Ask about the best time to visit, flights,
        hotels, local events, visa requirements, and more.<br><br>
        To get started, tell Marco your name.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="name-section">
        <h3>Who's travelling today?</h3>
        <p>Marco will remember your conversation throughout your session.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        name_input = st.text_input("Your name", placeholder="e.g. Sarah, James, Amara...", label_visibility="collapsed")
    with col2:
        start = st.button("Start →")

    if start and name_input.strip():
        st.session_state.user_name = name_input.strip()
        st.rerun()
    elif start and not name_input.strip():
        st.warning("Please enter your name to continue.")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT — shown after name is entered
# ══════════════════════════════════════════════════════════════════════════════
else:
    user_name = st.session_state.user_name

    # ── Top bar ───────────────────────────────────────────────────────────────
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown(f'<div class="session-badge">🧳 Travelling as <strong>{user_name}</strong></div>', unsafe_allow_html=True)
    with col2:
        with st.container():
            st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
            if st.button("New trip"):
                st.session_state.user_name    = ""
                st.session_state.chat_history = []
                st.session_state.greeted      = False
                if user_name.lower().strip() in store:
                    del store[user_name.lower().strip()]
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Welcome message (shown once) ──────────────────────────────────────────
    if not st.session_state.greeted:
        welcome = f"Hey {user_name}! 🌍 Marco here — your travel guide with a serious passport stamp collection. Tell me where you're dreaming of going, and I'll tell you everything you need to know — best time to visit, flights, hotels, local spots, visa info, all of it. What destination is calling your name?"
        st.session_state.chat_history.append({"role": "assistant", "content": welcome})
        st.session_state.greeted = True

    # ── Render chat history ───────────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message("user" if msg["role"] == "user" else "assistant", avatar="🧳" if msg["role"] == "user" else "🌍"):
            st.markdown(msg["content"])

    # ── Chat input ────────────────────────────────────────────────────────────
    user_input = st.chat_input(f"Ask Marco anything about your trip, {user_name}...")

    if user_input:
        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧳"):
            st.markdown(user_input)

        # Get Marco's response
        with st.chat_message("assistant", avatar="🌍"):
            with st.spinner("Marco is thinking..."):
                response = ask_marco(user_input, user_name)
            st.markdown(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})
