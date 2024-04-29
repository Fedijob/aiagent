import os
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.readers.file import PDFReader
from llama_index.core import VectorStoreIndex
from dotenv import load_dotenv

load_dotenv()

def get_index(data,index_name):
    index = None
    if not os.path.exists(index_name):
     print("building index", index_name)
     index=VectorStoreIndex.from_documents(data, show_progress=True)
     index.storage_context.persist(persist_dir=index_name)
    else:
     index = load_index_from_storage(
         StorageContext.from_defaults(persist_dir=index_name) 
         )
     
    return index   

pdf_path =os.path.join("data", "major.pdf")
major_pdf = PDFReader().load_data(file=pdf_path)
major_index = get_index(major_pdf, "major")
major_engine = major_index.as_query_engine()
