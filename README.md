# Data Analysis Agent

A high-performance, multi-modal autonomous agent designed for professional data exploration, statistical modeling, and interactive visualization. The agent leverages a unified SQL, Python, and R engine to deliver executive-grade insights with absolute data precision.

## Key Capabilities

### 1. Multi-Modal Analytical Engine
Seamlessly switch between different analytical paradigms depending on your technical needs:
- SQL Mode: Direct database-style querying for high-speed aggregation.
- Python Mode: Advanced Machine Learning (AutoML) and deep statistical analysis using pandas, scikit-learn, and XGBoost.
- R Mode: Tidyverse-powered statistical modeling and data manipulation.
- Generate Graphs: A three-step linear pipeline (Extraction -> Architect -> Coder) for creating high-quality interactive visualizations.
- Ask Questions: A ReAct-based conversational interface for ad-hoc data inquiries.

### 2. High-Precision Graph Pipeline
- SQL-Driven Extraction: Visualizations are backed by a hidden SQL aggregation engine to ensure 100% accurate metrics.
- Metadata Parity: Axis labels, legends, and tooltips are automatically synced with your dataset headers.
- Interactive HTML: Export charts as standalone interactive files for executive presentations.

### 3. Executive Reporting and Layman Insights
- Structured Summaries: Results are formatted as professional markdown reports with clean headers and tabular data.
- Layman Translations: Technical metrics (like Accuracy or R2) are automatically translated into plain-English sentences for non-technical stakeholders.
- Key Insights: Every analysis concludes with 3-4 bulleted takeaways explaining the real-world impact of the data.

### 4. Persistent Workspace
- Multi-Mode History: Your chat history, CSV results, and interactive graphs persist across mode switches, allowing you to build a comprehensive research thread without losing previous work.
- Auto-Snapshots: Every dataset upload is automatically summarized with AI-driven onboarding reports and visual data previews.

## How to Use the Agent

Follow these steps to perform a successful data analysis session:

1. Setup Your Data:
   - Use the sidebar to upload your CSV or Excel files. 
   - Once uploaded, the agent will automatically analyze the structure and provide a "Data Onboarding Report" in the chat.

2. Select the Appropriate Mode:
   - SQL Code: Use this if you want to perform quick filters, joins, or aggregations (e.g., "Find top 10 customers by spend").
   - Python Code: Use this for complex data cleaning, feature engineering, or machine learning (e.g., "Predict if a customer will churn").
   - Generate Graphs: Use this when you want a visual representation of your data (e.g., "Plot monthly revenue growth").
   - Ask Questions: Use this for direct natural language answers or complex statistical tests (e.g., "Is there a significant difference between Group A and Group B?").

3. Input Your Prompt:
   - Type your request in natural language in the chat box at the bottom.
   - For SQL, Python, or R modes, the agent will generate the code first. Review it, then click the "Perform Operation and Generate CSV" button to execute it.

4. Download and Export:
   - For data results, use the "Download Results (CSV)" button that appears after execution.
   - For visualizations, use the "Download Chart" button to save the interactive HTML file.

## Use Case Examples

| Mode | What you enter | What you get |
| :--- | :--- | :--- |
| SQL Code | "Show me total revenue by month" | Valid SQL code, CSV result preview, and Download button |
| Python Code | "Build a model to predict churn" | Robust ML code, Layman evaluation, and Predictive CSV |
| R Code | "Run a linear regression on price" | Clean R tidyverse code and Statistical summary |
| Generate Graphs | "Plot sales vs marketing spend" | High-quality interactive Plotly/Chart.js visualization |
| Ask Questions | "What is the average age in table X?" | Direct answer and Markdown tables for structured lists |

## Technical Setup

### Prerequisites
- Python 3.9 or higher
- An NVIDIA API Key or OpenRouter API Key (configured in environment or Streamlit secrets)

### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
Run the following command in your terminal:
```bash
streamlit run Data_Analysis_Agent.py
```

---
Built for data-driven precision and executive clarity.
