# Data Analytics Assistant

## Overview
The **Data Analytics Assistant** is a sophisticated, AI-powered web application designed to democratize data analysis by enabling users to explore and visualize datasets through natural language queries. Built with advanced machine learning and a user-friendly interface, this tool empowers both technical and non-technical users to derive actionable insights from CSV data files without requiring expertise in programming or database management. It seamlessly integrates natural language processing, SQL query generation, and dynamic visualization capabilities, making it an invaluable asset for data analysts, business professionals, and decision-makers.

## Features
- **Natural Language Querying**: Users can interact with their data by typing questions in plain English (e.g., "What is the average revenue by region?" or "Plot a bar chart of movie ratings from 1994"). The system leverages Azure's ChatOpenAI to interpret queries and execute corresponding actions.
- **SQL Query Generation**: For complex analytical questions, the tool automatically converts natural language into precise SQL queries, executed via the `pandasql` library. This allows users to perform advanced data filtering and aggregation without writing SQL code (e.g., "Which products had sales above average in Q1?").
- **Dynamic Visualizations**: The application generates high-quality visualizations using Matplotlib and Seaborn, supporting charts like bar graphs, histograms, and scatter plots. Visualizations are rendered instantly and available for download as PNG files.
- **Intelligent Question Suggestions**: Based on the dataset's structure and content, the tool generates tailored example questions to guide users in exploring their data effectively.
- **Robust Data Handling**: The system gracefully manages missing values, diverse data types, and large datasets, ensuring reliable analysis and visualization outputs.
- **Secure Code Execution**: Visualization code is executed in a controlled environment with strict safety checks to prevent execution of malicious or unsafe operations.

## Technical Architecture
- **Frontend**: A responsive web interface powered by Streamlit, providing an intuitive platform for file uploads, query input, and result display.
- **Backend**:
  - **AI Engine**: Utilizes AzureChatOpenAI for natural language processing, query interpretation, and code generation.
  - **Data Processing**: Leverages `pandas` for efficient DataFrame operations and `pandasql` for SQL query execution.
  - **Visualization**: Employs `matplotlib` and `seaborn` for generating visualizations, with direct Python execution for optimized performance.
  - **Image Handling**: Uses `PIL` (Pillow) to validate and process visualization outputs.
- **Security**: Implements code sanitization to block unsafe operations (e.g., `os.system`, `exec`, `eval`) and restricts execution globals to essential libraries.

## Installation
### Prerequisites
- Python 3.8 or higher
- Required Python packages (install via `requirements.txt`):
  ```bash
  pip install streamlit pandas pandasql matplotlib seaborn pillow langchain langchain-experimental langchain-community openai
  ```
- Azure OpenAI API credentials (configured via Streamlit secrets)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/data-analytics-assistant.git
   cd data-analytics-assistant
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure secrets in a `.streamlit/secrets.toml` file:
   ```toml
   AZURE_DEPLOYMENT = "your-deployment-name"
   AZURE_ENDPOINT = "your-azure-endpoint"
   AZURE_API_KEY = "your-api-key"
   AZURE_API_VERSION = "your-api-version"
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage
1. **Launch the Application**: Start the app and access it via a web browser (typically at `http://localhost:8501`).
2. **Select Query Type**: Choose between "Normal (CSV)" for general analysis and visualization or "SQL" for SQL-based queries.
3. **Upload Data**: Upload a CSV file containing your dataset.
4. **Process Data**: Click "Process Data" to load the dataset and initialize the AI agent.
5. **Interact with Data**:
   - For Normal mode, type questions or visualization requests (e.g., "Plot a histogram of sales" or "What are the top 5 products by revenue?").
   - For SQL mode, enter natural language questions to generate and execute SQL queries (e.g., "Show average order value by customer segment").
6. **Explore Results**: View data summaries, interactive visualizations, or SQL query results. Download visualizations as PNG files for reports or presentations.
7. **Use Example Questions**: Leverage AI-generated example questions in Normal mode to discover insights tailored to your dataset.

## Example Workflow
- **Dataset**: A CSV file with movie data (columns: `startYear`, `primaryTitle`, `averageRating`).
- **Query**: "Plot a bar chart of average ratings for movies from 1994."
- **Result**: The tool filters the data, generates a horizontal bar chart, and displays it with a download option.
- **SQL Query**: "Which genres had the highest average rating in the 1990s?"
- **Result**: The system converts the question into a SQL query, executes it, and presents a table of results.

## Limitations
- **SQL Mode**: Requires the `pandasql` library for SQL query execution. If not installed, SQL functionality is disabled.
- **Data Size**: Performance may vary with very large datasets due to in-memory processing.
- **Visualization Scope**: Limited to Matplotlib and Seaborn capabilities; complex interactive visualizations (e.g., D3.js) are not supported.
- **Security**: While safety checks are in place, users should avoid executing untrusted code in production environments.

## Future Enhancements
- Support for additional file formats (e.g., Excel, JSON).
- Integration of interactive visualization libraries (e.g., Plotly).
- Enhanced SQL query optimization for larger datasets.
- Multi-language support for natural language queries beyond English and European languages.
- Batch query processing for automated report generation.

## Contributing
Contributions are welcome! Please submit issues or pull requests via the GitHub repository. Ensure code adheres to PEP 8 standards and includes appropriate tests.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For questions or support, contact the project maintainer at [your-email@example.com] or open an issue on GitHub.

---

*Last Updated: April 15, 2025*
