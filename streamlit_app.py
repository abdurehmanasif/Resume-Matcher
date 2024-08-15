import streamlit as st
import faiss
import pandas as pd
import json
import re
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load the FAISS index and resumes
index_path = './vectorstore/index.faiss'  # Update this with your index path
index = faiss.read_index(index_path)

FAISS_PATH = "vectorstore"  # Path to the folder containing index.pkl and index.faiss
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Load the FAISS index with dangerous deserialization allowed
faiss_index = FAISS.load_local(
    FAISS_PATH,
    HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": 'cpu'}),
    allow_dangerous_deserialization=True
)

# Load the CSV file containing resumes into a DataFrame
resumes_df = pd.read_csv('output.csv', header=None, names=['ID', 'Resume'])

# Streamlit app title and welcome message
st.title("Professional Resume Matcher")
with st.sidebar:
    st.subheader("Instructions")
    st.write("1. Enter your OpenAI API key in the sidebar.")
    st.write("2. Enter the job description in the main area.")
    st.write("3. Click the 'Find Matching Resumes' button.")

    # Input for OpenAI API key
    api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    
    # Input for number of results
    num_results = st.number_input("Enter Number of Results:", min_value=1, value=5, step=1)

# Job description input
job_description = st.text_area("Enter Job Description:", height=200,placeholder="e.g., Looking for a Python developer with experience in Flask and REST APIs.")

if st.button("Find Matching Resumes"):
    if not api_key:
        st.error("Please enter your OpenAI API key.")
    else:
        # Initialize the ChatOpenAI model with the provided API key
        turbo_llm = ChatOpenAI(
            temperature=0,
            model_name='gpt-4o-mini',
            openai_api_key=api_key
        )

        # Create the RetrievalQAChain to answer questions
        qa_chain = RetrievalQA.from_chain_type(
            llm=turbo_llm, 
            chain_type="stuff", 
            retriever=faiss_index.as_retriever(), 
            return_source_documents=True
        )

        # Define a function to process the LLM response
        def process_llm_response(llm_response):
            result = llm_response['result']
            sources = llm_response["source_documents"]
            
            # Remove leading and trailing whitespace
            result = result.strip()
            
            # Replace single quotes with double quotes
            result = result.replace("'", '"')
            
            return result, sources

        # Define a prompt template for extracting information from the text
        review_template = """\
            For the following text, extract the following information and return it in the format:
            Name: [Name]
            Email: [Email]
            Phone: [Phone]
            Skills: [Skill1, Skill2, Skill3]
            Education: [Degree, Institution]
            Projects: [Project1, Project2]
            Work Experience: [Job Title, Company, Duration]

            Text: {text}
            """

        prompt_template = ChatPromptTemplate.from_template(review_template)

        # Process each resume and extract information
        try:
            results = faiss_index.similarity_search_with_relevance_scores(job_description, k=num_results)

            # Prepare to extract data for each matched resume
            candidates = []
            for res, score in results:
                filename = res.metadata['ID']
                resume_content = resumes_df.loc[resumes_df['ID'] == filename, 'Resume'].values
                if resume_content.size > 0:
                    resume_text = resume_content[0]
                    st.write(f"Processing resume: {filename}")

                    # Use the prompt template to create the query
                    query = prompt_template.format(text=resume_text)

                    # Call the QA chain to extract information
                    llm_response = qa_chain(query)
                    result, sources = process_llm_response(llm_response)

                    # Log the raw response for troubleshooting

                    # Use regex to extract information
                    info = {}
                    # Extract Name
                    name_match = re.search(r'Name:\s*(.*)', result)
                    if name_match:
                        info['Name'] = name_match.group(1).strip()

                    # Extract Email
                    email_match = re.search(r'Email:\s*(.*)', result)
                    if email_match:
                        info['Email'] = email_match.group(1).strip()

                    # Extract Phone (optional)
                    phone_match = re.search(r'Phone:\s*(.*)', result)
                    if phone_match:
                        info['Phone'] = phone_match.group(1).strip()
                    else:
                        info['Phone'] = "N/A"

                    # Extract Skills
                    skills_match = re.search(r'Skills:\s*\[(.*)\]', result)
                    if skills_match:
                        info['Skills'] = [skill.strip() for skill in skills_match.group(1).split(',')]
                    else:
                        info['Skills'] = []

                    # Extract Education
                    education_match = re.search(r'Education:\s*\[(.*)\]', result)
                    if education_match:
                        info['Education'] = [edu.strip() for edu in education_match.group(1).split(',')]
                    else:
                        info['Education'] = []

                    # Extract Projects
                    projects_match = re.search(r'Projects:\s*\[(.*)\]', result)
                    if projects_match:
                        info['Projects'] = [proj.strip() for proj in projects_match.group(1).split(',')]
                    else:
                        info['Projects'] = []

                    # Extract Work Experience
                    work_exp_match = re.search(r'Work Experience:\s*\[(.*)\]', result)
                    if work_exp_match:
                        info['Work Experience'] = [exp.strip() for exp in work_exp_match.group(1).split(',')]
                    else:
                        info['Work Experience'] = []

                    # Append extracted information if valid
                    if 'Name' in info and info['Name']:
                        candidates.append({
                            "Name": info['Name'],
                            "Email": info.get("Email", "N/A"),
                            "Resume Score": score,
                            "Key Skills": ", ".join(info.get("Skills", [])),
                            "Years of Experience": ", ".join(info.get("Work Experience", [])),
                            "Qualifications": ", ".join(info.get("Education", [])),
                            "Phone": info.get("Phone", "N/A"),
                        })

            # Display extracted information in a table
            if candidates:
                st.subheader("Matching Candidates")
                candidate_df = pd.DataFrame(candidates)

                # Display the DataFrame using st.table
                st.dataframe(candidate_df[['Name', 'Resume Score', 'Phone','Email', 'Key Skills', 'Years of Experience', 'Qualifications']])
            else:
                st.write("No matching candidates found.")        
        except Exception as e:
            st.error(f"An error occurred: {e}")