from langchain.chains.question_answering import load_qa_chain, LLMChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
import csv
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import re
import os


def output_to_dict(str):
    # Convert JSON String to Python
    json_output = json.loads(extract_first_substring(str))
    return json_output


def extract_first_substring(input_string):
    start_idx = input_string.find('{')
    end_idx = input_string.find('}')
    return input_string[start_idx: end_idx + 1]


def summerised_text(text, d_q):
    messages = [
        SystemMessage(
            content="you are a chemistry professor. I will provide to you academic paper in chemistry "
                    "and a list of guiding questions \n" \
                    f"please summarise  the paper. \n" \
                    f"the summary should focus on the questions list.\n" \
                    "please keep in the summary all important details to answer the questions like "
                    "numbers, parameter, procedures, conclusions and all that is needed to answer the questions"
                    "your summary should help other students to prepare for exam about the paper and they should be "
                    "able to master all the details of this paper based on your summery"),
        HumanMessage(content=f"questions: {d_q}, paper in chemistry: {text} "),
    ]

    chat = ChatOpenAI(openai_api_key="",
                      model_name='gpt-4o', temperature=0.0)
    result = chat.invoke(messages)

    return result.content


def count_tokens(docs):
    num_tokens = 0
    for doc in docs:
        num_tokens += len(doc.page_content.split(' '))
    return num_tokens


def extract_paper_no_references(pdf_path):
    # load pdf
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f'Number of tokens before delete references: {count_tokens(docs)}')
    new_docs = []
    for doc in docs:
        doc_lower_case = doc.page_content.lower()
        if 'references' in doc_lower_case:
            index_references = doc_lower_case.find('references')
            doc.page_content = doc_lower_case[:index_references]
            new_docs.append(doc)
            break
        else:
            new_docs.append(doc)

    print(f'Number of tokens after delete references: {count_tokens(new_docs)}')

    return new_docs


questions_dict = {"q1": "Please mention the names of the authors of this article",
                  "q2": "Mention the affiliation of the authors of this article",
                  "q3": "Which is the publishers house for this article?",
                  "q4": "What is the name of the journal where this article is published?",
                  "q5": "What is the title of the article?",
                  "q6": "What type of article isit? Is it a review, research, or another category altogether?",
                  "q7": "What is the year of publication for this article?",
                  "q8": "Does this article have any supporting information?",
                  "q9": "About which halide perovskite is mentioned in this article?",
                  "q10": "Did the halide perovskite mentioned in this article show photoluminescence?",
                  "q11": "Please provide the specific excitation and emission wavelengths for the halide perovskite "
                         "mentioned in the article.",
                  "q12": "What characterization techniques are used to analyze the material mentioned in this article?",
                  "q13": "Go to the experimental section and give the names of metal precursors and their "
                         "concentration used to synthesize perovskite mentioned in the article",
                  "q14": "What is the value of PLQY (Photoluminescence quantum yields) or QY (quantum yields) of the "
                         "halide perovskite mentioned",
                  "q15": "What is the material used for photocatalysis that is mentioned in this article?",
                  "q16": "Mention the major application of the material mentioned in this article. "
                         "photocatalysis mentioned?",
                  "q17": "What is the amount (mg) of catalyst used for photocatalysis?",
                  "q18": "What is the power in watts of the lamp used for photocatalysis?",
                  "q19": "What solar spectrum condition is used in the photocatalysis mentioned",
                  "q20": "What light intensity is used in the illumination for photocatalysis"}

questions_dict_sec_round = {"q1": "Please mention the names of the authors of this article.",
                            "q2": "Mention the affiliation of the authors of this article.",
                            "q3": "Which is the publisher's house for this article?",
                            "q4": "What is the name of the journal where this article is published?",
                            "q5": "What is the title of the article?",
                            "q6": "What type of article is it? Is it a review, research, or another category "
                                  "altogether?",
                            "q7": "What is the year of publication for this article?",
                            "q8": "Does this article have any supporting information?",
                            "q9": "About which halide perovskite is mentioned in this article?",
                            "q10": "What is the excitation and emission wavelength for the halide perovskite "
                                   "mentioned in the "
                                   "article? First mention excitation and emission separated by a comma without units.",
                            "q11": "Name the characterization techniques in abbreviations used to analyze the material "
                                   "mentioned in this article. Separate with commons.",
                            "q12": "Go to the experimental section and give the names of metal precursors and their "
                                   "concentration used to synthesize perovskite mentioned in the article.",
                            "q13": "What is the material used for photocatalysis that is mentioned in this article?",
                            "q14": "Mention the major application of the material mentioned in this article. Which "
                                   "type of reaction was it, photovoltaic or photocatalytic?",
                            "q15": "What is the amount (mg) of catalyst used for photocatalysis?",
                            "q16": "What is the power of the lamp in watts, light intensity in mW/cm 2 , and the "
                                   "solar spectrum condition used in the photocatalysis? Provide the answers in "
                                   "numbers without units separated by common.",
                            "q17": "What is the highest rate of electron consumption (μmol g−1) for the composites "
                                   "reported in the article? Mention the rate of electron consumption without units.",
                            "q18": "What is the duration of photocatalysis mentioned in the article in hours? Mention "
                                   "the duration without units.",
                            "q19": "What solvent system is used for the photocatalysis mentioned in the article?",
                            "q20": "What is the maximum yield in μmol g−1 of the produced CO in the photocatalysis "
                                   "mentioned in this article without units? Mention the yield without unit.",
                            "q21": "What is the maximum yield of the CH4 in μmol g−1 produced during photocatalysis "
                                   "mentioned in the article? Mention the yield without units.",
                            "q22": "Which synthesis method was used to synthesize halide perovskite, 1) hot injection "
                                   "method 2) Colloidal chemistry, 3) antisolvent precipitation, or 4) none of the "
                                   "above? Provide Answers in the option number",
                            "q23": "What is the dimension of the NCs in nm mentioned in the article? Answer only in "
                                   "numbers without unit. "
                            }


def save_str_to_text_file(input_str, output_file_name, method):
    with open(f"outputs/{output_file_name}_{method}.txt", "w") as text_file:
        text_file.write(input_str)
        text_file.close()


def save_dict_to_csv(input_dict, output_file_name, method):
    # Get the keys and values from the dictionary
    if not os.path.exists(f'/Users/kfirraiby/Desktop/git/chemistry_QA/outputs/{method}'):
        os.makedirs(f'/Users/kfirraiby/Desktop/git/chemistry_QA/outputs/{method}')

    keys = list(input_dict.keys())
    values = list(input_dict.values())

    # Write to CSV file
    with open(f"outputs/{method}/{output_file_name}_{method}.csv", 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write header (keys)
        csv_writer.writerow(['Keys', 'Values'])

        # Write data (keys and values)
        for key, value in zip(keys, values):
            csv_writer.writerow([key, value])


def list_files_in_folder(folder_path, file_type='pdf'):
    try:
        # Get the list of files in the folder
        files = os.listdir(folder_path)

        # Filter out subdirectories, if any
        files = [file for file in files if file.lower().endswith(f".{file_type}")]

        return files
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def save_summary_as_textfile(summary, pdf_name, folder_path):
    # Create a new file name with '_summary' suffix
    base_name = os.path.splitext(pdf_name)[0]
    summary_file_name = f"{base_name}_summary.txt"
    summary_file_path = os.path.join(folder_path, summary_file_name)

    # Save the summary text to the file
    with open(summary_file_path, 'w', encoding='utf-8') as file:
        file.write(summary)