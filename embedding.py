from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
output_dir = r"C:\Users\owner\Desktop\Projects\nlp"
embeddings.save_pretrained(output_dir)