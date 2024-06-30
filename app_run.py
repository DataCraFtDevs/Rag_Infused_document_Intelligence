import os
import pandas as pd
import streamlit as st
from langchain_community.llms import Ollama
from language_model import process_single_document
from process_Model import process_multiple_documents
from interface_app import process_pdfs

# Main Streamlit app code
def main():
    col1, col2, col3 = st.columns([1, 2, 5])

    with col1:
        st.image("/Users/admin/Documents/workspace/venv/Rag_Automation/RAG_prototype/VIRIDIEN_Logo.png", width=700)

    with col2:
        st.write("")  # Empty space to align with the title

    with col3:
        st.markdown("")

    st.title("RAG-Infused Document Intelligence: Version_1")

    # Sidebar input for folder path and user prompt
    folder_path = st.text_input("Enter the folder path containing PDF files:")
    user_prompt = st.text_input("Enter prompt for language model:")

    # Dropdown for selecting model type
    model_type = st.selectbox("Select Model Type", ["llama3", "mistral"])

    # Upload CSV or Excel file for data types (tags)
    uploaded_file = st.file_uploader("Upload CSV or Excel file for data types (tags)", type=["csv", "xlsx"])

    # Dropdown for selecting word count
    word_count_options = ["1", "2"]  # Add more options as needed
    word_count = int(st.selectbox("Select number of tags to respond with", word_count_options))

    file_name = st.text_input("Enter file name for saving results (without extension):", "results")

    # Process button
    if st.button("Process PDFs"):
        if folder_path:
            if os.path.isdir(folder_path):
                data_types = []

                # Load data types (tags) from uploaded file (CSV or Excel)
                if uploaded_file is not None:
                    if uploaded_file.name.endswith('.csv'):
                        data_types_df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith('.xlsx'):
                        data_types_df = pd.read_excel(uploaded_file)
                    
                    # Check if the 'tag' column exists and let the user select if it doesn't
                    if "tag" in data_types_df.columns:
                        data_types = data_types_df["tag"].tolist()
                    else:
                        st.warning("The uploaded file does not contain a 'tag' column. Please select the correct column.")
                        column_options = data_types_df.columns.tolist()
                        selected_column = st.selectbox("Select the column containing tags", column_options)
                        if selected_column:
                            data_types = data_types_df[selected_column].tolist()

                # Process PDFs and get results based on selected model type
                results_df = process_pdfs(folder_path, user_prompt, model_type, data_types, word_count)

                # Create a dedicated folder for saving results
                save_folder = "results"
                os.makedirs(save_folder, exist_ok=True)

                # Save results as CSV in the dedicated folder
                save_path = os.path.join(save_folder, f"{file_name}.csv")
                results_df.to_csv(save_path, index=False)

                # Display results
                st.success(f"Results saved successfully at {save_path}")
                st.dataframe(results_df)
            else:
                st.error(f"'{folder_path}' is not a valid directory.")
        else:
            st.warning("Please enter a folder path.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
