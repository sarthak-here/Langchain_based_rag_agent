"""
Interactive CLI for the RAG agent.
Run: python cli.py
"""

from rag_agent import RAGAgent

HELP_TEXT = """
Commands:
  reset        — Clear conversation memory
  sources on   — Show source documents with each answer
  sources off  — Hide source documents
  help         — Show this message
  exit / quit  — Exit the CLI
"""


def main():
    print("=" * 60)
    print("  LangChain RAG Agent — CLI")
    print("  Type 'help' for available commands.")
    print("=" * 60)

    agent = RAGAgent()
    show_sources = False

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

        if user_input.lower() == "help":
            print(HELP_TEXT)
            continue

        if user_input.lower() == "sources on":
            show_sources = True
            print("Sources display: ON")
            continue

        if user_input.lower() == "sources off":
            show_sources = False
            print("Sources display: OFF")
            continue

        if show_sources:
            response, sources = agent.chat_with_sources(user_input)
            print(f"\nAssistant: {response}")
            if sources:
                print("\n  Sources:")
                for src in sources:
                    page_info = f", page {src['page']}" if src.get("page") is not None else ""
                    print(f"    - {src['source']}{page_info}")
                    print(f"      \"{src['snippet']}\"")
        else:
            response = agent.chat(user_input)
            print(f"\nAssistant: {response}")


if __name__ == "__main__":
    main()
