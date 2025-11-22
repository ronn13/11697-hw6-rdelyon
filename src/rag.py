drive_root_folder = '/content/gdrive/My Drive/Colab Notebooks/Data/RAG_documents'

import os
from langchain_core.documents import Document # Moved import here

os.chdir(drive_root_folder)
fileChunks = []
files = os.listdir()
print(files)
for file in files:
    with open(file, 'r') as f:
        document = f.read()
        fileChunks.append(Document(page_content=document))