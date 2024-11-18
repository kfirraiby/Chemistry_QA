from langchain_community.document_loaders import PyPDFLoader
import csv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os

questions_dict = {"q1": "Please mention the names of the authors of this article. ",
                  "q2": "What is the name of the journal where this article is published?",
                  "q3": "What is the title of the article?",
                  "q4": "What type of article is it? Is it a review, research, or another category altogether?",
                  "q5": "What is the year of publication for this article?",
                  "q6": "About which halide perovskite is mentioned in this article?",
                  "q7": "What is the excitation and emission wavelength for the halide perovskite mentioned in the "
                        "article? First mention excitation and emission separated by a comma without units.",
                  "q8": "Go to the experimental section and give the names of metal precursors and their "
                        "concentration used to synthesize perovskite mentioned in the article.",
                  "q9": "What is the material used for photocatalysis that is mentioned in this article?",
                  "q10": "Mention the major application of the material mentioned in this article."
                         "Which type of reaction was it, photovoltaic or photocatalytic?",
                  "q11": "(a) What is the amount (mg) of catalyst used for photocatalysis?, (b) the concentration of "
                         "the catalyst used.",
                  "q12": "What is the power of the lamp in watts, light intensity in mW/cm2, and the solar spectrum "
                         "condition used in the photocatalysis? Provide the answers in numbers without units "
                         "separated by common. ",
                  "q13": "What is the highest rate of electron consumption (μmol g−1) for the composites reported in "
                         "the article? Mention the rate of electron consumption without units. ",
                  "q14": "What is the duration of photocatalysis mentioned in the article in hours? Mention the "
                         "duration without units.",
                  "q15": "(a)What solvent system is used for the photocatalysis mentioned in the article?"
                        "(b) The amount of solvent used for the photocatalysis",
                  "q16": "How long is the system degassed or purged with inert gas before initiating the "
                         "photocatalysis?",
                  "q17": "What is the rate and total time used for CO₂ purging to achieve saturated CO₂ conditions?",
                  "q18": "For this photocatalytic system, what are the major products obtained from the CO₂ reduction?",
                  "q19": "What is the maximum yield or production rate in μmol g−1 of the produced CO in the "
                         "photocatalysis mentioned in this article without units? Mention the yield without unit.",
                  "q20": "What is the maximum yield or production rate of the CH4 in μmol g−1 produced during "
                         "photocatalysis mentioned in the article? Mention the yield without units.",
                  "q21": "What is the apparent quantum efficiency of the system for the photocatalytic CO2 reduction?",
                  "q22": "How many times greater is the formation rate of each product in the composites compared to "
                         "the rate achieved with pristine CsPbBr₃?",
                  "q23": "What is the selectivity of the system for CO₂ reduction to each product?",
                  "q24": "Which synthesis method was used to synthesize halide perovskite, 1) hot injection method 2) "
                         "Colloidal chemistry, 3) antisolvent precipitation, or 4) none of the above? Provide Answers "
                         "in the option number",
                  "q25": "What is the dimension of the NCs in nm mentioned in the article? Answer only in numbers "
                         "without unit.",
                  }


def summerised_text(text):
    messages = [
        SystemMessage(
            content="you are a chemistry professor. I will provide to you academic paper in chemistry "
                    "and a list of guiding questions \n" \
                    f"please summarise  the paper. \n" \
                    f"the summary should focus on the questions list.\n" \
                    "please keep in the summary all important details to answer the questions like "
                    "numbers/parameter/procedures/conclusions and all that is needed to answer the questions) "
                    "your summary should help other students to prepare for exam about the paper and they should be "
                    "able to master all the details of this paper based on your summery"),
        HumanMessage(content=f"questions: {questions_dict}, paper in chemistry: {text} "),
    ]

    chat = ChatOpenAI(openai_api_key="",
                      model_name='gpt-4-1106-preview', temperature=0.0)
    result = chat.invoke(messages)

    return result.content


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


def save_dict_to_csv(input_dict,path, output_file_name, method):
    # Get the keys and values from the dictionary
    if not os.path.exists(f'{path}/{method}'):
        os.makedirs(f'{path}/{method}') ############## you are here #########

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


def count_tokens(docs):
    num_tokens = 0
    for doc in docs:
        num_tokens += len(doc.page_content.split(' '))
    return num_tokens
