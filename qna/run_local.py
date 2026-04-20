from config import DATA_FOLDER
from config import MODEL_NAME
from qa_agent import FinancialAnalyticsAgent

print("Loading data...")

agent = FinancialAnalyticsAgent(data_folder=DATA_FOLDER, model_name=MODEL_NAME)
agent.bootstrap()

print("\nQ&A Agent Ready")
print("Type 'exit' to quit")

while True:

    q = input("\nAsk question: ")

    if q.lower() == "exit":
        break

    answer = agent.ask(q)

    print("\nAnswer:")
    print(answer)