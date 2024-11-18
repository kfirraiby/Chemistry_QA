from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from utils import summerised_text, questions_dict, save_dict_to_csv, extract_paper_no_references, list_files_in_folder, \
    questions_dict_sec_round, save_summary_as_textfile
import os
import time
papers_folder_name = 'test_papers'
os.environ["OPENAI_API_KEY"] = ""
folder_path = f"/Users/kfirraiby/Desktop/git/chemistry_QA/{papers_folder_name}"
file_list = list_files_in_folder(folder_path)

# read the paper with no references.
total_time = 0
for pdf in file_list:
    start_time = time.time()
    paper_path = folder_path + '/' + pdf
    paper = [extract_paper_no_references(paper_path)]
    print(paper)

    #  get the paper as a full string
    input_text = ' '
    for i in paper[0]:
        input_text += i.page_content

    # summarized text using CHAT GPT focused on the questions as guidance
    sum_text = summerised_text(text=input_text, d_q=questions_dict_sec_round)
    save_summary_as_textfile(sum_text, pdf, folder_path)

    print(f"num_token after summary: {len(sum_text.split(' '))}")

#     # loop over the questions, input the q and the summarized text
#     all_QA_dict = {}
#     for q in questions_dict_sec_round:
#         messages = [
#             SystemMessage(
#                 content="you are a chemistry professor. I will provide to you a summary of academic paper in "
#                         "chemistry and  questions "
#                         "about it.\n"
#                         "please answer the question with short and  accurate answers as you can.\n"
#                         "base your answers using only the provided text"
#                         "if there is no answer for any reason just output the word None "
#                         "for example if the answer is a number/parameter please provide only the number and units of "
#                         "scale."
#                         "another example: if the answer is a chemistry process please provide only the professional "
#                         "name of the process "
#                         "another example: if the answer has few params or numbers please prvide the answer as follows "
#                         "1.parma, 2.param .. "
#                         "please answer with professional  and academic terms only.\n"
#                         "please provide your output in the format as follows: Keys,Values where keys are the question number and values are your answer "),
#             HumanMessage(content=f"question: {questions_dict_sec_round[q]}, paper in chemistry: {sum_text}"),
#         ]
#
#         chat = ChatOpenAI(openai_api_key="",
#                           model_name='gpt-4o',
#                           temperature=0.0)
#
#         result = chat.invoke(messages)
#         all_QA_dict[q] = result.content
#
#     save_dict_to_csv(all_QA_dict, output_file_name=pdf.rstrip('.pdf') + '_' + 'output', method='post_summery_sec_round_gpt_4o_v2')
#     end_time = time.time()
#     iteration_time = end_time - start_time
#     total_time += iteration_time
#     print(f"Iteration {pdf}: Time taken - {iteration_time:.4f} seconds")
#
#
# print(total_time)
