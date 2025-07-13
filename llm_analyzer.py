import os
import json
from typing import Dict, List, Optional, Tuple
import streamlit as st
from google import genai
from dotenv import load_dotenv

load_dotenv()

class LLMAnalyzer:
    """Uses LLM to analyze user prompts and generate chart specifications."""
    
    def __init__(self):
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = None
        self.model = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-001')
        self.temperature = float(os.getenv('TEMPERATURE', '0.7'))
        self.max_tokens = int(os.getenv('MAX_TOKENS', '2000'))
    
    def analyze_prompt(self, user_prompt: str, data_info: Dict) -> Dict:
        """
        Analyze user prompt and generate chart specifications.
        
        Args:
            user_prompt: User's natural language prompt
            data_info: Information about the loaded dataset
            
        Returns:
            Dict: Chart specifications including chart type, columns, and parameters
        """
        try:
            if not self.client:
                st.error("Gemini API client not initialized. Please check your API key.")
                return self._get_default_chart_spec(data_info)
            
            # Create a comprehensive prompt for the LLM
            system_prompt = self._create_system_prompt(data_info)
            
            # Prepare the content for the model
            content = f"{system_prompt}\n\nUser request: {user_prompt}"
            
            # Generate response using the new API
            response = self.client.models.generate_content(
                model=self.model,
                contents=content,
                config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens
                }
            )
            
            # Parse the response
            chart_spec = self._parse_llm_response(response.text)
            return chart_spec
            
        except Exception as e:
            st.error(f"Error analyzing prompt with LLM: {str(e)}")
            return self._get_default_chart_spec(data_info)
    
    def _create_system_prompt(self, data_info: Dict) -> str:
        """Create a system prompt with data context for the LLM."""
        columns_info = []
        for col in data_info.get('columns', []):
            dtype = data_info.get('dtypes', {}).get(col, 'unknown')
            missing = data_info.get('missing_values', {}).get(col, 0)
            columns_info.append(f"- {col} ({dtype}, {missing} missing values)")
        
        prompt = f"""
You are an expert data analyst and visualization specialist. Your task is to analyze user requests and generate specifications for creating charts from structured data.

Available data columns:
{chr(10).join(columns_info)}

Data shape: {data_info.get('shape', 'Unknown')}

Available chart types:
1. bar - For comparing categories or showing distributions
2. line - For time series or continuous relationships
3. scatter - For correlation analysis between two numeric variables
4. pie - For showing proportions of a whole
5. histogram - For showing distribution of a single variable
6. box - For showing distribution and outliers
7. heatmap - For correlation matrices or 2D categorical data
8. area - For showing cumulative values over time
9. bubble - For three-dimensional data (x, y, size)
10. violin - For showing distribution shape

Your response should be a valid JSON object with the following structure:
{{
    "chart_type": "chart_type_name",
    "title": "Descriptive title for the chart",
    "x_axis": "column_name_for_x_axis",
    "y_axis": "column_name_for_y_axis",
    "color": "column_name_for_color_encoding",
    "size": "column_name_for_size_encoding",
    "facet": "column_name_for_faceting",
    "aggregation": "sum|mean|count|max|min",
    "sort_by": "column_name_to_sort_by",
    "limit": number_of_items_to_show,
    "orientation": "vertical|horizontal",
    "description": "Brief description of what the chart shows",
    "insights": ["insight1", "insight2", "insight3"]
}}

Rules:
1. Choose the most appropriate chart type based on the user's request
2. Use ONLY columns that exist in the data - DO NOT invent column names
3. For numeric columns, suggest aggregations when appropriate
4. For categorical columns, consider limiting results to top N items
5. Provide meaningful titles and descriptions
6. Suggest relevant insights that could be drawn from the chart
7. If the request is unclear, ask for clarification or suggest a default chart
8. If color/size/facet columns are requested but not available, omit them from the JSON
9. Double-check that all column names exactly match the available columns
10. For bar charts with multiple categories, use color encoding to show sub-categories (e.g., "sales by region, product" should use product as color)
11. When user mentions multiple dimensions (e.g., "by X, Y"), consider using Y as color encoding for better visualization

Respond only with the JSON object, no additional text.
"""
        return prompt
    
    def _parse_llm_response(self, response: str) -> Dict:
        """Parse the LLM response and extract chart specifications."""
        try:
            # Clean the response to extract JSON
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            
            chart_spec = json.loads(response)
            return chart_spec
            
        except json.JSONDecodeError as e:
            st.warning(f"Could not parse LLM response as JSON: {str(e)}")
            return {}
    
    def _get_default_chart_spec(self, data_info: Dict) -> Dict:
        """Generate a default chart specification when LLM fails."""
        numeric_cols = data_info.get('numeric_columns', [])
        categorical_cols = data_info.get('categorical_columns', [])
        
        if len(numeric_cols) >= 2:
            return {
                "chart_type": "scatter",
                "title": f"Correlation between {numeric_cols[0]} and {numeric_cols[1]}",
                "x_axis": numeric_cols[0],
                "y_axis": numeric_cols[1],
                "description": f"Scatter plot showing the relationship between {numeric_cols[0]} and {numeric_cols[1]}",
                "insights": ["Shows correlation between variables", "Identifies potential outliers", "Reveals data patterns"]
            }
        elif len(categorical_cols) > 0 and len(numeric_cols) > 0:
            return {
                "chart_type": "bar",
                "title": f"{numeric_cols[0]} by {categorical_cols[0]}",
                "x_axis": categorical_cols[0],
                "y_axis": numeric_cols[0],
                "aggregation": "mean",
                "description": f"Bar chart showing average {numeric_cols[0]} for each {categorical_cols[0]}",
                "insights": ["Compares values across categories", "Shows distribution patterns", "Highlights top performers"]
            }
        else:
            return {
                "chart_type": "bar",
                "title": "Data Overview",
                "description": "Default chart showing data distribution",
                "insights": ["Basic data visualization", "Shows data structure", "Provides overview of dataset"]
            }
    
    def generate_chart_description(self, chart_spec: Dict, data_summary: Dict) -> str:
        """Generate a natural language description of the chart."""
        try:
            prompt = f"""
Based on the following chart specification and data summary, generate a clear, informative description of what the chart will show:

Chart Specification:
{json.dumps(chart_spec, indent=2)}

Data Summary:
- Total rows: {data_summary.get('total_rows', 'Unknown')}
- Total columns: {data_summary.get('total_columns', 'Unknown')}

Generate a 2-3 sentence description that explains:
1. What type of chart will be created
2. What data will be visualized
3. What insights can be expected

Keep the description clear and accessible to non-technical users.
"""
            
            # Generate response using the new API
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": 0.5,
                    "max_output_tokens": 150
                }
            )
            
            return response.text.strip()
            
        except Exception as e:
            return f"This chart will visualize {chart_spec.get('title', 'the data')} to help understand patterns and relationships in your dataset."
    
    def suggest_alternative_charts(self, data_info: Dict) -> List[Dict]:
        """Suggest alternative chart types based on the data."""
        suggestions = []
        
        numeric_cols = data_info.get('numeric_columns', [])
        categorical_cols = data_info.get('categorical_columns', [])
        datetime_cols = data_info.get('datetime_columns', [])
        
        # Time series chart
        if len(datetime_cols) > 0 and len(numeric_cols) > 0:
            suggestions.append({
                "chart_type": "line",
                "title": f"{numeric_cols[0]} over time",
                "description": f"Line chart showing {numeric_cols[0]} trends over {datetime_cols[0]}"
            })
        
        # Distribution chart
        if len(numeric_cols) > 0:
            suggestions.append({
                "chart_type": "histogram",
                "title": f"Distribution of {numeric_cols[0]}",
                "description": f"Histogram showing the distribution of {numeric_cols[0]}"
            })
        
        # Category comparison
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            suggestions.append({
                "chart_type": "bar",
                "title": f"Top categories by {numeric_cols[0]}",
                "description": f"Bar chart comparing {numeric_cols[0]} across {categorical_cols[0]} categories"
            })
        
        # Correlation analysis
        if len(numeric_cols) >= 2:
            suggestions.append({
                "chart_type": "scatter",
                "title": f"Correlation: {numeric_cols[0]} vs {numeric_cols[1]}",
                "description": f"Scatter plot showing relationship between {numeric_cols[0]} and {numeric_cols[1]}"
            })
        
        return suggestions[:5]  # Limit to 5 suggestions 