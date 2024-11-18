import os
import hashlib
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from final_sys_utils import extract_paper_no_references, summerised_text, questions_dict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


# Step 1: Load existing CSV database (if it exists)
def load_database(csv_file):
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        # Ensure the 'identifier' column exists
        if 'identifier' not in df.columns:
            df['identifier'] = pd.Series(dtype='str')
    else:
        # Create an empty DataFrame with an 'identifier' column
        df = pd.DataFrame(columns=['identifier'])
    return df


# Step 2: Extract the headline from the PDF
def extract_pdf_headline(pdf_file):
    loader = PyPDFLoader(pdf_file)
    pages = loader.load()

    # Assuming the headline is in the first page's content
    first_page_content = pages[0].page_content.strip()

    # For simplicity, let's assume the headline is the first line of the first page
    headline = first_page_content.split('\n')[0]

    return headline


# Step 3: Create a unique identifier using the headline of the PDF
def generate_pdf_identifier(pdf_file):
    headline = extract_pdf_headline(pdf_file)
    return hashlib.md5(headline.encode()).hexdigest(), headline


# Step 4: Process the PDF and get the dictionary from LLM
def process_pdf(pdf_file):
    # Example: Return dictionary from LLM output after processing the PDF
    paper = [extract_paper_no_references(pdf_file)]
    input_text = ' '
    for i in paper[0]:
        input_text += i.page_content

    # summarized text using CHAT GPT focused on the questions as guidance
    sum_text = summerised_text(input_text)

    all_QA_dict = {}
    for q in questions_dict:
        messages = [
            SystemMessage(
                content="you are a chemistry professor. I will provide to you a summary of academic paper in chemistry a question "
                        "about it.\n"
                        "please answer the question with short and  accurate answers as you can.\n"
                        "base your answers using only the provided text"
                        "if there is no answer for any reason just output the word None "
                        "for example if the answer is a number/parameter please provide only the number and units of scale." \
                        "another example: if the answer is a chemstry prosess plaese provide only the profetional name of the process"
                        "another example: if the answer has few params or numbers please prvide the answer as follows 1. "
                        "parma, 2.param .. "
                        "please answer with professional  and academic terms only.\n"
                        f"please provide your putput as follows: your answer "),
            HumanMessage(content=f"question: {questions_dict[q]}, paper in chemistry: {sum_text}"),
        ]

        chat = ChatOpenAI(openai_api_key="",
                          model_name='gpt-4-1106-preview',
                          temperature=0.0)

        result = chat.invoke(messages)
        all_QA_dict[q] = result.content

    return {'Summary text': sum_text, **all_QA_dict}  # Replace with actual LLM output


# Step 5: Check if the new entry already exists in the database
def is_entry_in_database(df, identifier):
    return identifier in df['identifier'].values


# Step 6: Append new data to the database if not a duplicate
def update_database(df, new_data, identifier, headline):
    # Add identifier to the new data
    new_data['identifier'] = identifier
    new_data['name'] = headline
    # Convert new_data (a dictionary) to a DataFrame before concatenation
    # new_row = pd.DataFrame([new_data])
    # Concatenate the new row to the existing DataFrame
    df = pd.concat([df, new_data], ignore_index=True)
    return df


# Step 7: Save the updated database to CSV
def save_database(df, csv_file):
    df.to_csv(csv_file, index=False)


# Main program to handle multiple files
def main(directory, csv_file):
    # Load the existing database
    df = load_database(csv_file)

    # Iterate over all PDF files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_file = os.path.join(directory, filename)

            # Generate a unique identifier for the PDF based on the headline
            identifier, headline = generate_pdf_identifier(pdf_file)

            # Check if the PDF was already processed
            if is_entry_in_database(df, identifier):
                print(f"{pdf_file} has already been processed. Skipping...")
            else:
                # Process the PDF and get LLM output
                new_data = pd.DataFrame([process_pdf(pdf_file)])

                # Update the database with the new data
                df = update_database(df, new_data, identifier, headline)

                # Save the updated database after each file to ensure progress is saved
                save_database(df, csv_file)
                print(f"Data from {pdf_file} has been added to the database.")


# Example usage
if __name__ == "__main__":
    directory = "/Users/kfirraiby/Desktop/git/chemistry_QA/test_papers"  # Directory containing PDF files
    csv_file = "/Users/kfirraiby/Desktop/git/chemistry_QA/final_database/database.csv"
    main(directory, csv_file)
