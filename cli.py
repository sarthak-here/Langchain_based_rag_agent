"""
Interactive CLI for the RAG agent.
Run: python cli.py
"""

from rag_agent import RAGAgent


def main():
    print("=" * 60)
    print("  LangChain RAG Agent — CLI")
    print("  Type 'exit' or 'quit' to stop.")
    print("  Type 'reset' to clear conversation memory.")
    print("=" * 60)

    agent = RAGAgent()

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if user_input.lower() == "reset":
            agent.reset_memory()
            continue

        response = agent.chat(user_input)
        print(f"\nAssistant: {response}")


if __name__ == "__main__":
    main()
