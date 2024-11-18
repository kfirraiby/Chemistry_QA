from langchain.chains.question_answering import load_qa_chain
from utils import extract_paper_no_references, questions_dict, output_to_dict, save_str_to_text_file, files_paths_dict, save_dict_to_csv, questions_dict_sec_round
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

import os

os.environ["OPENAI_API_KEY"] = "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"

name_paper = 'AI paper new'
file_path = files_paths_dict[name_paper]

paper = [extract_paper_no_references(file_path)]

input_text = ' '
for i in paper[0]:
    input_text += i.page_content

all_QA_dict = {}

for q in questions_dict_sec_round:
    messages = [
        SystemMessage(
            content="you are a chemistry professor. I will provide to you an academic paper in chemistry a question "
                    "about it.\n"
                    "please answer the question with short and  accurate answers as you can.\n"
                    "base your answers using only the provided text"
                    "if there is no answer for any reason just output the word 'None' "
                    "for example if the answer is a number/parameter please provide only the number and units of scale."\
                    "another example: if the answer is a chemstry prosess plaese provide only the profetional name of "
                    "the process" 
                    "another example: if the answer has few params or numbers please prvide the answer as follows 1. "
                    "parma, 2.param .. "
                    "please answer with professional  and academic terms only.\n"
                    f"please provide your putput as follows: 'your answer' "),
        HumanMessage(content=f"questions: {questions_dict[q]}, paper in chemistry: {input_text} "),
    ]

    chat = ChatOpenAI(openai_api_key="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", model_name='gpt-4-1106'
                                                                                                       '-preview',temperature=0.0)

    result = chat.invoke(messages)

    # results_dict = output_to_dict(result.content)
    print(q)
    print(result.content)
    all_QA_dict[q] = result.content

print(all_QA_dict)

# save_dict_to_csv(all_QA_dict, output_file_name=name_paper + '_' + 'output', method='full')