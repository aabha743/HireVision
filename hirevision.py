import PyPDF2
from langchain.llms import Ollama


#You can replace the model to any other model that you have installed via Ollama
llm_pipeline = Ollama(base_url="http://localhost:11434", model="falcon3")

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = "".join(page.extract_text() for page in reader.pages)
    return text

def get_llm_response(job_description, resume_text):
    #Send JD and resume text to the LLM and return the response.
    prompt = (
        "As a recruiter, your task is to match the provided Job Description with the candidate's Resume, "
        "rank the candidate's suitability for the role on a scale from 0 to 10, and provide a brief explanation "
        "justifying the score based on the alignment of skills, experience, and qualifications.\n"
        f"Job Description: {job_description}\n"
        f"Resume: {resume_text}"
    )
    response = llm_pipeline(prompt)  
    return response


#Incase you want to just test whetehr the model you are trying to run is working of not, un-comment the below lines and run them to check for any errors
#text = extract_text_from_pdf(r"C:\Users\aabha\Downloads\Devansh_Soni_s_CV.pdf")
#jd = "We are seeking a skilled and innovative AI Engineer to join our dynamic team. The ideal candidate will be responsible for designing, developing, and implementing AI models and algorithms to solve complex business problems. You will collaborate closely with data scientists, software engineers, and product teams to integrate machine learning solutions into production systems. Key responsibilities include building and optimizing machine learning models, conducting experiments to improve model performance, and deploying AI-driven applications. Strong expertise in programming languages such as Python, deep learning frameworks like TensorFlow or PyTorch, and a solid understanding of data structures, algorithms, and cloud platforms is essential. The ideal candidate should also have excellent problem-solving skills, a passion for AI technologies, and a collaborative mindset."
#response = get_llm_response(jd,text)
#print(response)