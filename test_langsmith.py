from dotenv import load_dotenv
from langsmith import Client

load_dotenv()
client = Client()

projects = list(client.list_projects())
print(" Connected! Projects:", [p.name for p in projects])