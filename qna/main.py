from config import DATA_FOLDER
from config import MODEL_NAME
from qa_agent import FinancialAnalyticsAgent

agent = FinancialAnalyticsAgent(data_folder=DATA_FOLDER, model_name=MODEL_NAME)
agent.bootstrap()

while True:

    q = input("\nAsk question: ")

    if q == "exit":
        break

    print("\n", agent.ask(q))