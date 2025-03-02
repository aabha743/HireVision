import PyPDF2
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer, util
import torch

# Load LLM and embedding model
llm_pipeline = OllamaLLM(base_url="http://localhost:11434", model="llama3.1")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Fast & lightweight

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    reader = PyPDF2.PdfReader(pdf_file)
    text = "\n".join([page.extract_text() or "" for page in reader.pages]).strip()
    return text if text else "No text extracted from PDF."

def compute_cosine_similarity(job_description, resume_text):
    """Compute cosine similarity between JD and Resume using BERT embeddings."""
    if not job_description or not resume_text:
        return 0.0  # Return 0 similarity for empty inputs
    
    jd_embedding = embedding_model.encode(job_description, convert_to_tensor=True)
    resume_embedding = embedding_model.encode(resume_text, convert_to_tensor=True)
    
    similarity_score = util.pytorch_cos_sim(jd_embedding, resume_embedding).squeeze().item()
    print(round(similarity_score,2))
    return round(similarity_score, 2)

def get_llm_response(job_description, resume_text, similarity_score):
    """Generate LLM-based suitability score with explanation, considering cosine similarity."""
    prompt = (
        "As a recruiter, analyze the provided Job Description and Resume.\n"
        f"Cosine similarity score: {similarity_score} (0 to 1, higher means better match).\n"
        "Now, rate the candidate's suitability for this role on a scale from 0 to 10 and provide an explanation "
        "justifying the score based on skills, experience, and qualifications.\n\n"
        f"Job Description:\n{job_description}\n\n"
        f"Resume:\n{resume_text}"
    )
    
    response = llm_pipeline.invoke(prompt)  # FIX: Use invoke() instead of calling LLM directly
    return response

# Example usage:
# resume_text = extract_text_from_pdf("resume.pdf")
# jd_text = "Your job description here"
# similarity = compute_cosine_similarity(jd_text, resume_text)
# llm_feedback = get_llm_response(jd_text, resume_text, similarity)
# print("LLM Feedback:", llm_feedback)
import PyPDF2
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer, util
import torch

# Load LLM and embedding model
llm_pipeline = OllamaLLM(base_url="http://localhost:11434", model="llama3.1")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Fast & lightweight

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    reader = PyPDF2.PdfReader(pdf_file)
    text = "\n".join([page.extract_text() or "" for page in reader.pages]).strip()
    return text if text else "No text extracted from PDF."

def compute_cosine_similarity(job_description, resume_text):
    """Compute cosine similarity between JD and Resume using BERT embeddings."""
    if not job_description or not resume_text:
        return 0.0  # Return 0 similarity for empty inputs
    
    jd_embedding = embedding_model.encode(job_description, convert_to_tensor=True)
    resume_embedding = embedding_model.encode(resume_text, convert_to_tensor=True)
    
    similarity_score = util.pytorch_cos_sim(jd_embedding, resume_embedding).squeeze().item()
    print(round(similarity_score,2))
    return round(similarity_score, 2)

def get_llm_response(job_description, resume_text, similarity_score):
    """Generate LLM-based suitability score with explanation, considering cosine similarity."""
    prompt = (
        "As a recruiter, analyze the provided Job Description and Resume.\n"
        f"Cosine similarity score: {similarity_score} (0 to 1, higher means better match).\n"
        "Now, rate the candidate's suitability for this role on a scale from 0 to 10 and provide an explanation "
        "justifying the score based on skills, experience, and qualifications.\n\n"
        f"Job Description:\n{job_description}\n\n"
        f"Resume:\n{resume_text}"
    )
    
    response = llm_pipeline.invoke(prompt)  # FIX: Use invoke() instead of calling LLM directly
    return response

# Example usage:
# resume_text = extract_text_from_pdf("resume.pdf")
# jd_text = "Your job description here"
# similarity = compute_cosine_similarity(jd_text, resume_text)
# llm_feedback = get_llm_response(jd_text, resume_text, similarity)
# print("LLM Feedback:", llm_feedback)
