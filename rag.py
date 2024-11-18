import os

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import FAISS
import time


from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from utils import questions_dict , save_dict_to_csv, list_files_in_folder
from utils import extract_paper_no_references

from collections import defaultdict

os.environ["OPENAI_API_KEY"] = ""

folder_path = "/Users/kfirraiby/Desktop/git/chemistry_QA/test_papers"
file_list = list_files_in_folder(folder_path)

total_time = 0
for pdf in file_list:
    start_time = time.time()
    paper_path = folder_path + '/' + pdf
    paper = [extract_paper_no_references(paper_path)]
    document = paper[0]
    # print(type(document))
    # print(document)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=1000)
    texts = text_splitter.split_documents(document)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)

    retriever = db.as_retriever(search_kwargs={"k": 5})  #
    # docs = retriever.get_relevant_documents("What is the power in watts of the lamp used for photocatalysis?")
    all_QA_dict = {}

    system_template = """the following context is parts from an academic paper in chemestry.
                        you are a chemistry professor. I will provide  parts from an academic paper in chemistry as context.\n"
                        "please answer the question with short and  accurate answers as you can.\n"
                        "base your answers using only the provided context"
                        "if there is no answer for any reason just output the word 'None' "
                        "for example if the answer is a number/parameter please provide only the number and units of scale." \
                        "another example: if the answer is a chemstry prosess plaese provide only the profetional name of the process"
                        "another example: if the answer has few params or numbers please prvide the answer as follows 1. "
                        "parma, 2.param .. "
                        "please answer with professional  and academic terms only.\n"
                        f"please provide your output as follows: 'your answer'
                        Use the following pieces of context to answer the user's question.
                        If there is no answer in the context, just output the word 'None', don't try to make up an answer.
    ----------------
    {context}"""

    human_template = "{question}"

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template),
    ]

    qa_prompt = ChatPromptTemplate.from_messages(messages)

    all_QA_dict = {}
    for q in questions_dict:
        query = questions_dict[q]
        retrived_docs = retriever.get_relevant_documents(query)

        qa = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model_name='gpt-4-1106-preview', temperature=0.0), chain_type="stuff", retriever=retriever,
                return_source_documents=True, chain_type_kwargs={"prompt": qa_prompt})
        #
        result = qa.invoke({"query": query})['result']
        #     results_dict[questions_dict[q_num]] = {result['result']: str(result['source_documents'])}
        all_QA_dict[q] = result

    # save_dict_to_csv(all_QA_dict, output_file_name=pdf.rstrip('.pdf') + '_' + 'output', method='rag_chunk4k_lap1k_k5')
    end_time = time.time()
    iteration_time = end_time - start_time
    total_time += iteration_time
    print(f"Iteration {pdf}: Time taken - {iteration_time:.4f} seconds")

print(total_time)
