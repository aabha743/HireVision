# HireVision

Welcome to **HireVision**, a powerful tool designed to help job seekers assess how well their resumes align with job descriptions (JDs). This project extracts key information from your resume and compares it to the job description you're applying for, providing a suitability score from 0 to 10, along with a brief explanation justifying the score.

The goal of HireVision is to help candidates understand how their qualifications match the job's expectations and improve their chances of landing an interview.

## Features

- **Resume and Job Description Matching**: Upload your resume as a PDF and paste the job description for the role you are applying for.
- **Text Extraction**: Resume text is extracted automatically using the `PyPDF2` library.
- **LLM Evaluation**: A large language model (LLM) analyzes the resume and job description, providing a suitability score.
- **Suitability Score & Explanation**: A suitability score (0-10) and a justification of the score are generated based on the match between the resume and job description.
- **User-Friendly Interface**: Simple interface for easy interaction, allowing you to upload your resume and input the job description.

## Requirements

Before running the project, ensure that you have the following libraries installed:

- Python 3.x
- `PyPDF2` for PDF text extraction
- Ollama and the llama3.1 model or any other model of your liking, but make sure you change the model you've used in the hirevision.py file.
- Streamlit for the interface
- LangChain , as we are utilising the model downloaded via Ollama through LangChain.
- Sentence Transformers library to calculate the cosine scores.

Install the required Python dependencies via `pip` after cloning the repository:

```bash
pip install -r requirements.txt


