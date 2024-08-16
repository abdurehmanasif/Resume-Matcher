import streamlit as st
import faiss
import pandas as pd
import re
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load the FAISS index and resumes
FAISS_PATH = "vectorstore"  # Path to the folder containing index.pkl and index.faiss
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Load the FAISS index
faiss_index = FAISS.load_local(
    FAISS_PATH,
    HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": 'cpu'}),
    allow_dangerous_deserialization=True
)

# Load resumes into a DataFrame
resumes_df = pd.read_csv('output.csv', header=None, names=['ID', 'Resume'])

# Initialize session state to store previous inputs and outputs
if 'search_count' not in st.session_state:
    st.session_state.search_count = 0
if 'searches' not in st.session_state:
    st.session_state.searches = []

# Streamlit app title and sidebar instructions
st.title("Professional Resume Matcher")
with st.sidebar:
    st.subheader("Instructions")
    st.write("1. Enter your OpenAI API key.")
    st.write("2. Enter the job description.")
    st.write("3. Click 'Find Matching Resumes'.")

    api_key = st.text_input("OpenAI API Key:", type="password")
    num_results = st.number_input("Number of Results:", min_value=1, value=5)

def search_and_extract(job_description, api_key, num_results):
    """Perform a search and extract relevant resume information."""
    # Initialize the ChatOpenAI model
    turbo_llm = ChatOpenAI(
        temperature=0,
        model_name='gpt-4o-mini',
        openai_api_key=api_key
    )

    # Create the RetrievalQAChain
    qa_chain = RetrievalQA.from_chain_type(
        llm=turbo_llm, 
        chain_type="stuff", 
        retriever=faiss_index.as_retriever(), 
        return_source_documents=True
    )

    # Define a prompt template for extracting information
    review_template = """\
        For the following text, extract the following information:
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
    candidates = []
    try:
        results = faiss_index.similarity_search_with_relevance_scores(job_description, k=num_results)
        for res, score in results:
            filename = res.metadata['ID']
            resume_content = resumes_df.loc[resumes_df['ID'] == filename, 'Resume'].values
            if resume_content.size > 0:
                resume_text = resume_content[0]
                st.write(f"Processing resume: {filename}")

                # Create the query and call the QA chain
                query = prompt_template.format(text=resume_text)
                llm_response = qa_chain(query)

                # Extract information using regex
                result = llm_response['result'].strip().replace("'", '"')
                info = {
                    'Name': re.search(r'Name:\s*(.*)', result).group(1).strip() if re.search(r'Name:\s*(.*)', result) else "N/A",
                    'Email': re.search(r'Email:\s*(.*)', result).group(1).strip() if re.search(r'Email:\s*(.*)', result) else "N/A",
                    'Phone': re.search(r'Phone:\s*(.*)', result).group(1).strip() if re.search(r'Phone:\s*(.*)', result) else "N/A",
                    'Skills': [skill.strip() for skill in re.search(r'Skills:\s*\[(.*)\]', result).group(1).split(',')] if re.search(r'Skills:\s*\[(.*)\]', result) else [],
                    'Education': [edu.strip() for edu in re.search(r'Education:\s*\[(.*)\]', result).group(1).split(',')] if re.search(r'Education:\s*\[(.*)\]', result) else [],
                    'Projects': [proj.strip() for proj in re.search(r'Projects:\s*\[(.*)\]', result).group(1).split(',')] if re.search(r'Projects:\s*\[(.*)\]', result) else [],
                    'Work Experience': [exp.strip() for exp in re.search(r'Work Experience:\s*\[(.*)\]', result).group(1).split(',')] if re.search(r'Work Experience:\s*\[(.*)\]', result) else []
                }

                # Append extracted information if valid
                if info['Name'] != "N/A":
                    candidates.append({
                        "Name": info['Name'],
                        "Email": info['Email'],
                        "Resume Score": score,
                        "Key Skills": ", ".join(info['Skills']),
                        "Years of Experience": ", ".join(info['Work Experience']),
                        "Qualifications": ", ".join(info['Education']),
                        "Phone": info['Phone'],
                    })

        return candidates
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Display previous searches
for i, search in enumerate(st.session_state.searches):
    st.subheader(f"Search {i+1}")
    st.write(f"**Job Description:**\n{search['job_description']}")
    if search['candidates']:
        st.dataframe(pd.DataFrame(search['candidates']))
    else:
        st.write("No matching candidates found.")

# Job description input for a new search
st.subheader("New Search")
job_description = st.text_area("Job Description:", height=200, key=f"job_description_{st.session_state.search_count}")

if st.button("Find Matching Resumes", key=f"find_button_{st.session_state.search_count}"):
    if not api_key:
        st.error("Please enter your OpenAI API key.")
    elif not job_description.strip():
        st.error("Please enter a job description.")
    else:
        # Perform search and extract information
        candidates = search_and_extract(job_description, api_key, num_results)

        if candidates is not None:
            # Store the search results in session state
            st.session_state.searches.append({
                'job_description': job_description,
                'candidates': candidates if candidates else None
            })

            # Increment the search count
            st.session_state.search_count += 1

            # Display the current search results
            st.write(f"**Search {st.session_state.search_count}:**")
            st.write(f"**Job Description:**\n{job_description}")
            if candidates:
                st.dataframe(pd.DataFrame(candidates))
            else:
                st.write("No matching candidates found.")

# Display new input and button below the most recent search
if st.session_state.search_count > 0:
    st.subheader("Continue Searching")
    job_description = st.text_area("Job Description:", height=200, key=f"job_description_{st.session_state.search_count}")

    if st.button("Find Matching Resumes", key=f"find_button_{st.session_state.search_count}"):
        candidates = search_and_extract(job_description, api_key, num_results)

        if candidates is not None:
            st.session_state.searches.append({
                'job_description': job_description,
                'candidates': candidates if candidates else None
            })
            st.session_state.search_count += 1

            st.write(f"**Search {st.session_state.search_count}:**")
            st.write(f"**Job Description:**\n{job_description}")
            if candidates:
                st.dataframe(pd.DataFrame(candidates))
            else:
                st.write("No matching candidates found.")
