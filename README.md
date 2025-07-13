# Natural Language Analytics Dashboard

An intelligent analytics dashboard that uses LLM (Large Language Model) to automatically generate charts and visualizations from structured data based on natural language prompts.

## 🌐 Live Demo

**Try the application online:** [https://subhammoda-natural-language-analytics-dashboard.streamlit.app/](https://subhammoda-natural-language-analytics-dashboard.streamlit.app/)

## Features

- **File Upload**: Support for CSV and Excel files
- **Natural Language Processing**: Describe what you want to see in plain English
- **Automatic Chart Generation**: AI-powered chart selection and creation
- **Data Analysis**: Automatic data insights and recommendations
- **Export Capabilities**: Download generated charts and reports
- **Sample Data**: Built-in sample data for quick testing
- **Chart History**: Track and regenerate previous charts

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Gemini API key (for LLM functionality)

## Installation

### Step 1: Clone or Download the Project

```bash
git clone <repository-url>
cd natural-language-analytics-dashboard
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

If you encounter any issues, you can install packages individually:

```bash
pip install streamlit pandas numpy plotly matplotlib seaborn google-genai python-dotenv openpyxl xlrd langchain langchain-google-genai python-multipart
```

### Step 3: Set Up Environment Variables

1. Create a `.env` file in the project root directory:
```bash
touch .env
```

2. Add your Gemini API key to the `.env` file:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

3. (Optional) Configure additional settings:
```
GEMINI_MODEL=gemini-2.5-flash
TEMPERATURE=0.7
MAX_TOKENS=2000
DEBUG=False
LOG_LEVEL=INFO
```

### Step 4: Run the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Usage

1. **Upload Data**: Use the file uploader to upload a CSV or Excel file, or click "Load Sample Data" to try with sample data.

2. **Describe Your Chart**: In the text area, describe what kind of chart you want to see. For example:
   - "Show me a bar chart of sales by region"
   - "Create a line chart showing revenue trends over time"
   - "Display a pie chart of product categories"

3. **Generate**: Click the "Generate Chart" button to create your visualization.

4. **Explore**: Try different prompts and explore the generated charts and insights.

## Example Prompts

- "Show me a bar chart of sales by region"
- "Create a line chart showing revenue trends over time"
- "Display a pie chart of product categories"
- "Show correlation between price and sales volume"
- "Create a heatmap of customer satisfaction scores"
- "Show distribution of sales by product type"

## Project Structure

```
natural-language-analytics-dashboard/
├── app.py                 # Main Streamlit application
├── data_processor.py      # Data loading and preprocessing
├── chart_generator.py     # Chart generation logic
├── llm_analyzer.py        # LLM integration for analysis
├── utils.py               # Utility functions
├── requirements.txt       # Python dependencies
├── Chocolate Sales.csv    # Sample data for testing
└── README.md              # Project documentation
```

## Technologies Used

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **Google Gemini**: Natural language processing
- **LangChain**: LLM orchestration

## Troubleshooting

### Common Issues

#### 1. Import Errors
If you see import errors, make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

#### 2. Gemini API Key Issues
- Ensure your API key is correctly set in the `.env` file
- Check that you have sufficient credits in your Google AI Studio account
- Verify the API key format

#### 3. Port Already in Use
If port 8501 is already in use, Streamlit will automatically use the next available port. Check the terminal output for the correct URL.

#### 4. File Upload Issues
- Ensure your CSV/Excel file is properly formatted
- Check that the file size is reasonable (< 100MB)
- Verify the file has at least 2 columns

#### 5. Chart Generation Errors
- Make sure your data has the appropriate column types for the requested chart
- Try simpler prompts if complex ones fail
- Check the data preview to understand your data structure

### Getting Help

If you encounter issues not covered here:

1. Check the terminal output for error messages
2. Verify your Python version: `python --version`
3. Ensure all dependencies are up to date: `pip install --upgrade -r requirements.txt`
4. Check the project documentation and code comments

## Advanced Configuration

### Customizing Chart Styles

You can modify the chart appearance by editing the `ChartGenerator` class in `chart_generator.py`:

```python
# Change default colors
self.color_palette = px.colors.qualitative.Set1

# Change default chart size
self.default_height = 800
self.default_width = 1000
```

### Adding New Chart Types

To add new chart types, extend the `ChartGenerator` class:

1. Add a new method like `_create_custom_chart()`
2. Update the `create_chart()` method to handle the new type
3. Update the LLM prompt in `llm_analyzer.py` to include the new type

### Environment Variables

You can customize the application behavior by setting these environment variables:

- `GEMINI_MODEL`: "gemini-2.5-flash"
- `TEMPERATURE`: Control creativity vs. consistency (0.0-1.0)
- `MAX_TOKENS`: Limit response length
- `DEBUG`: Enable debug mode for more verbose output

## Security Considerations

1. **API Key Security**: Never commit your `.env` file to version control
2. **Data Privacy**: The application processes data locally, but be careful with sensitive data
3. **Rate Limiting**: Be aware of Google AI Studio API rate limits and costs
4. **Input Validation**: The application includes basic input validation, but always verify results

## Performance Tips

1. **Large Datasets**: For datasets with >10,000 rows, consider sampling or aggregating data
2. **Multiple Charts**: Use the chart history feature to avoid regenerating the same charts
3. **Caching**: The application caches some results in session state for better performance

## Next Steps

Once you have the basic application running, you can:

1. **Customize the UI**: Modify the Streamlit interface in `app.py`
2. **Add Data Sources**: Extend `data_processor.py` to support more file formats
3. **Enhance Charts**: Add more chart types and customization options
4. **Deploy**: Deploy the application to Streamlit Cloud or other platforms
5. **Integrate**: Connect to databases or external APIs for real-time data

## License

MIT License 