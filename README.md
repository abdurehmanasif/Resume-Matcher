# Professional Resume Matcher

## Overview

The **Professional Resume Matcher** is a Streamlit application designed to match job descriptions with resumes. It utilizes advanced language processing techniques to extract relevant information from resumes and provide a list of candidates that best fit the job requirements. This app is particularly useful for recruiters, hiring managers, and job seekers looking to streamline the resume screening process.

## Features

- **Job Description Input**: Enter a job description to find matching resumes.
- **Resume Extraction**: Automatically extracts candidate information such as name, email, phone number, skills, education, projects, and work experience from resumes.
- **Customizable Results**: Users can specify the number of results to display.
- **User-Friendly Interface**: A clean and intuitive interface for easy navigation and interaction.

## Technologies Used

- **Streamlit**: For building the web application.
- **FAISS**: For efficient similarity search in high-dimensional spaces.
- **Pandas**: For data manipulation and analysis.
- **Langchain**: For language model integration and prompt management.
- **Hugging Face Transformers**: For embedding models used in the application.

## Prerequisites

Before running the application, ensure you have the following:

- Python 3.7 or higher
- A GitHub account (for deployment)
- An OpenAI API key (for language model access)

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/Resume-Matcher.git
   cd Resume-Matcher
   ```

2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Required Packages**:
   Create a `requirements.txt` file if it doesn't exist, and include the following dependencies:
   ```plaintext
   streamlit
   faiss-cpu
   pandas
   langchain
   langchain-huggingface
   ```
   Then run:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Model**:
   Ensure you have the necessary models and data files. This may include the FAISS index and the CSV file containing resumes. Place these files in the appropriate directory as specified in the code.

## Usage

1. **Run the Application**:
   Start the Streamlit application by running:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Enter Your OpenAI API Key**:
   In the sidebar, enter your OpenAI API key to enable the language model functionality.

3. **Input Job Description**:
   In the main area, enter the job description for the position you are hiring for.

4. **Search for Matching Resumes**:
   Click the "Find Matching Resumes" button to initiate the search. The app will process the resumes and display the candidates that match your criteria.

5. **View Results**:
   The results will be displayed in a table format, showing the relevant information extracted from each resume.

## Example

Here’s an example of how to use the app:

1. Enter a job description such as:
   ```
   Looking for a Python developer with experience in Flask and REST APIs.
   ```

2. Click the "Find Matching Resumes" button.

3. Review the list of candidates displayed, including their names, emails, phone numbers, skills, and qualifications.

## Troubleshooting

- **EOF Error**: If you encounter an EOF error during deployment, check the logs for specific issues. Ensure all dependencies are installed correctly and that your data files are accessible.
- **Invalid API Key**: Make sure your OpenAI API key is valid and has the necessary permissions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/) for providing a powerful framework for building web applications.
- [FAISS](https://faiss.ai/) for efficient similarity search.
- [Langchain](https://github.com/hwchase17/langchain) for integrating language models into applications.

---
