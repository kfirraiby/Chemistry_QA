from play.class_utils import extract_paper_no_references, questions_dict, from_docs_list_to_str, output_to_dict, \
    summerised_text
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


class PDFAnalyzer:

    def __init__(self, pdf_path, model):
        self.pdf_path = pdf_path
        self.documents = extract_paper_no_references(pdf_path)
        self.open_api_key = "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"
        self.model = model

    def full_text_method(self):
        input_text = from_docs_list_to_str(self.documents)

        messages = [
            SystemMessage(
                content="you are a chemistry professor. I will provide to you an academic paper in chemistry and "
                        "questions "
                        "about it.\n" \
                        "please answer all the questions with short and  accurate answers as you can.\n" \
                        "please answer with professional  and academic terms only.\n" \
                        "please provide your putput in a JSON format as follows { 'q1': 'your answer', 'q2': 'your "
                        "answer', "
                        "... }\n"
                        "please output only string jason '{...}" \
                        f"here are the question {str(questions_dict)}"),
            HumanMessage(content=f"questions: {questions_dict}, paper in chemistry: {input_text} "),
        ]

        chat = ChatOpenAI(openai_api_key=self.open_api_key,
                          model_name=self.model)

        result = chat.invoke(messages)

        results_dict = output_to_dict(result.content)

        return results_dict

    def post_summery_method(self, ):
        input_text = from_docs_list_to_str(self.documents)
        input_text = summerised_text(input_text)

        messages = [
            SystemMessage(
                content="you are a chemistry professor. I will provide to you with a summery of  academic paper in "
                        "chemistry "
                        "and "
                        "questions about it.\n" \
                        f"please answer all the questions with short and  accurate answers as you can.\n" \
                        f"please answer with professional  and academic terms only.\n" \
                        "please provide your output in a JSON format as follows {'q1': 'your answer', 'q2': 'your "
                        "answer', ... '}"),
            HumanMessage(content=f"questions: {questions_dict}, paper in chemistry: {input_text} "),
        ]

        chat = ChatOpenAI(openai_api_key=self.open_api_key,
                          model_name=self.model)

        result = chat.invoke(messages)

        results_dict = output_to_dict(result.content)

        return results_dict

    def run_analyze_pdf(self, method_choice, **params):

        if method_choice == 'full_text':
            return self.full_text_method(**params)
        elif method_choice == 'post_summery':
            return self.post_summery_method(**params)
        elif method_choice == 'rag':
            return self.rag_method(**params)
        else:
            print("Invalid method choice")
