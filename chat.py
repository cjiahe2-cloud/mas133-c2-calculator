from insurance_mas_rag.rag import Context, build_agent, latest_turn, load_env

load_env()

agent = build_agent()


def chat():
    config = {"configurable": {"thread_id": "session-1"}}
    ctx = Context(user_id="me")

    print("MAS notices assistant ready (MAS 125, MAS 133). Ask a question (type 'quit' to exit).\n")
    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            break

        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config=config,
            context=ctx,
        )
        reply, searches, _ = latest_turn(result["messages"])

        # Show what the agent searched for (helps you debug retrieval)
        if searches:
            print(f"  (searched: {'; '.join(searches)})")

        print(f"\nAssistant: {reply}\n")


if __name__ == "__main__":
    chat()
