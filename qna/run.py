import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_FOLDER
from config import MODEL_NAME
from qa_agent import FinancialAnalyticsAgent


def main() -> None:
    print("Loading aggregated JSON from data...")

    agent = FinancialAnalyticsAgent(data_folder=str(PROJECT_ROOT / DATA_FOLDER), model_name=MODEL_NAME)
    agent.bootstrap()

    print("\nQ&A Agent Ready")
    print("Type 'exit' to quit")

    while True:
        question = input("\nAsk question: ")
        if question.lower() == "exit":
            break

        answer = agent.ask(question)
        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()
