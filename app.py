import streamlit as st
import hirevision


st.title("HireVision: Job Fit Evaluator")

st.header("Upload Job Description and Resume")
job_description = st.text_area("Job Description", placeholder="Paste the job description here...")
resume_file = st.file_uploader("Upload Resume (PDF format)", type="pdf")

if st.button("Evaluate"):
    if not job_description or not resume_file:
        st.error("Please provide both the job description and a resume.")
    else:
        try:
            # Extract text from the uploaded resume PDF
            resume_text = hirevision.extract_text_from_pdf(resume_file)

            # Get LLM response
            llm_response = hirevision.get_llm_response(job_description, resume_text)

            # Display results
            st.subheader("Results")
            st.markdown(f"**Response:** {llm_response}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")