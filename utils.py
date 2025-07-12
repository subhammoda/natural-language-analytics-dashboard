import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, List, Optional, Tuple
import json
import io
import base64
from datetime import datetime

def validate_data(data: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate the uploaded data.
    
    Args:
        data: DataFrame to validate
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if data is None or data.empty:
        return False, "No data provided or data is empty."
    
    if len(data.columns) < 2:
        return False, "Data must have at least 2 columns for meaningful visualization."
    
    # Check for excessive missing values
    missing_percentages = data.isnull().sum() / len(data) * 100
    high_missing_cols = missing_percentages[missing_percentages > 50].index.tolist()
    
    if high_missing_cols:
        return False, f"Columns with >50% missing values: {', '.join(high_missing_cols)}"
    
    return True, "Data validation passed."

def format_number(value: float, decimals: int = 2) -> str:
    """Format a number for display."""
    if pd.isna(value):
        return "N/A"
    
    if abs(value) >= 1e6:
        return f"{value/1e6:.{decimals}f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.{decimals}f}K"
    else:
        return f"{value:.{decimals}f}"

def get_chart_download_link(fig: go.Figure, filename: str = "chart") -> str:
    """
    Generate a download link for a Plotly chart.
    
    Args:
        fig: Plotly figure object
        filename: Name for the downloaded file
        
    Returns:
        str: HTML download link
    """
    # Convert to PNG
    img_bytes = fig.to_image(format="png")
    
    # Create download link
    b64 = base64.b64encode(img_bytes).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}.png">Download PNG</a>'
    
    return href

def export_chart_data(data: pd.DataFrame, chart_spec: Dict) -> str:
    """
    Export chart data to CSV.
    
    Args:
        data: DataFrame used for the chart
        chart_spec: Chart specification
        
    Returns:
        str: CSV data as string
    """
    # Prepare data similar to chart generation
    x_axis = chart_spec.get('x_axis')
    y_axis = chart_spec.get('y_axis')
    aggregation = chart_spec.get('aggregation', 'sum')
    
    if x_axis and y_axis and x_axis in data.columns and y_axis in data.columns:
        if aggregation == 'count':
            export_data = data.groupby(x_axis).size().reset_index(name=y_axis)
        else:
            agg_func = getattr(data.groupby(x_axis)[y_axis], aggregation)
            export_data = agg_func().reset_index()
    else:
        export_data = data.copy()
    
    # Convert to CSV string
    csv_buffer = io.StringIO()
    export_data.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

def load_sample_data() -> pd.DataFrame:
    """Use Chocolate Sales Data for sample data."""
    
    data = pd.read_csv('./Chocolate Sales.csv')
    
    # Remove the first column
    data = data.iloc[:, 1:]
    
    return data

def get_chart_insights(data: pd.DataFrame, chart_spec: Dict) -> List[str]:
    """
    Generate insights about the chart data.
    
    Args:
        data: DataFrame used for the chart
        chart_spec: Chart specification
        
    Returns:
        List[str]: List of insights
    """
    insights = []
    
    try:
        x_axis = chart_spec.get('x_axis')
        y_axis = chart_spec.get('y_axis')
        chart_type = chart_spec.get('chart_type', 'bar')
        
        if x_axis and y_axis and x_axis in data.columns and y_axis in data.columns:
            # Basic statistics
            if data[y_axis].dtype in ['int64', 'float64']:
                mean_val = data[y_axis].mean()
                max_val = data[y_axis].max()
                min_val = data[y_axis].min()
                
                insights.append(f"Average {y_axis}: {format_number(mean_val)}")
                insights.append(f"Range: {format_number(min_val)} to {format_number(max_val)}")
            
            # Top performers
            if chart_type in ['bar', 'pie']:
                top_value = data[y_axis].max() if data[y_axis].dtype in ['int64', 'float64'] else None
                if top_value is not None:
                    top_category = data[data[y_axis] == top_value][x_axis].iloc[0]
                    insights.append(f"Highest {y_axis}: {top_category} ({format_number(top_value)})")
            
            # Distribution insights
            if chart_type == 'histogram':
                skewness = data[y_axis].skew()
                if abs(skewness) > 1:
                    direction = "right-skewed" if skewness > 0 else "left-skewed"
                    insights.append(f"Distribution is {direction}")
            
            # Correlation insights
            if chart_type == 'scatter':
                correlation = data[x_axis].corr(data[y_axis])
                if abs(correlation) > 0.7:
                    strength = "strong" if abs(correlation) > 0.8 else "moderate"
                    direction = "positive" if correlation > 0 else "negative"
                    insights.append(f"{strength} {direction} correlation ({correlation:.2f})")
        
        # Data quality insights
        missing_count = data.isnull().sum().sum()
        if missing_count > 0:
            insights.append(f"Contains {missing_count} missing values")
        
        insights.append(f"Total records: {len(data):,}")
        
    except Exception as e:
        insights.append("Unable to generate insights due to data processing error")
    
    return insights[:5]  # Limit to 5 insights

def create_chart_summary(chart_spec: Dict, data_summary: Dict) -> Dict:
    """
    Create a summary of the chart and data.
    
    Args:
        chart_spec: Chart specification
        data_summary: Data summary information
        
    Returns:
        Dict: Chart summary
    """
    return {
        "chart_type": chart_spec.get('chart_type', 'Unknown'),
        "title": chart_spec.get('title', 'Untitled Chart'),
        "description": chart_spec.get('description', 'No description available'),
        "data_shape": data_summary.get('shape', 'Unknown'),
        "columns_used": [
            col for col in [chart_spec.get('x_axis'), chart_spec.get('y_axis'), 
                           chart_spec.get('color'), chart_spec.get('size')] 
            if col is not None
        ],
        "aggregation": chart_spec.get('aggregation', 'None'),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def validate_chart_spec(chart_spec: Dict, data_info: Dict) -> Tuple[bool, str]:
    """
    Validate chart specification against available data.
    
    Args:
        chart_spec: Chart specification
        data_info: Data information
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not chart_spec:
        return False, "No chart specification provided"
    
    required_fields = ['chart_type', 'title']
    for field in required_fields:
        if field not in chart_spec:
            return False, f"Missing required field: {field}"
    
    # Validate column references
    available_columns = data_info.get('columns', [])
    column_fields = ['x_axis', 'y_axis', 'color', 'size', 'facet']
    
    for field in column_fields:
        if field in chart_spec and chart_spec[field]:
            if chart_spec[field] not in available_columns:
                return False, f"Column '{chart_spec[field]}' not found in data"
    
    # Validate chart type
    valid_chart_types = ['bar', 'line', 'scatter', 'pie', 'histogram', 'box', 'heatmap', 'area', 'bubble', 'violin']
    if chart_spec['chart_type'] not in valid_chart_types:
        return False, f"Invalid chart type: {chart_spec['chart_type']}"
    
    return True, "Chart specification is valid"

def format_chart_spec_for_display(chart_spec: Dict) -> str:
    """
    Format chart specification for display in the UI.
    
    Args:
        chart_spec: Chart specification
        
    Returns:
        str: Formatted specification
    """
    if not chart_spec:
        return "No specification available"
    
    formatted = []
    formatted.append(f"**Chart Type:** {chart_spec.get('chart_type', 'Unknown')}")
    formatted.append(f"**Title:** {chart_spec.get('title', 'Untitled')}")
    
    if chart_spec.get('x_axis'):
        formatted.append(f"**X-Axis:** {chart_spec['x_axis']}")
    if chart_spec.get('y_axis'):
        formatted.append(f"**Y-Axis:** {chart_spec['y_axis']}")
    if chart_spec.get('color'):
        formatted.append(f"**Color:** {chart_spec['color']}")
    if chart_spec.get('size'):
        formatted.append(f"**Size:** {chart_spec['size']}")
    if chart_spec.get('aggregation'):
        formatted.append(f"**Aggregation:** {chart_spec['aggregation']}")
    
    if chart_spec.get('description'):
        formatted.append(f"**Description:** {chart_spec['description']}")
    
    return "\n".join(formatted)

def get_chart_recommendations(data_info: Dict) -> List[Dict]:
    """
    Get chart recommendations based on data characteristics.
    
    Args:
        data_info: Data information
        
    Returns:
        List[Dict]: List of recommended charts
    """
    recommendations = []
    
    numeric_cols = data_info.get('numeric_columns', [])
    categorical_cols = data_info.get('categorical_columns', [])
    datetime_cols = data_info.get('datetime_columns', [])
    
    # Time series recommendation
    if datetime_cols and numeric_cols:
        recommendations.append({
            "type": "line",
            "title": f"Time Series: {numeric_cols[0]} over time",
            "description": f"Track {numeric_cols[0]} trends using {datetime_cols[0]}",
            "priority": "high"
        })
    
    # Distribution recommendation
    if numeric_cols:
        recommendations.append({
            "type": "histogram",
            "title": f"Distribution of {numeric_cols[0]}",
            "description": f"Understand the distribution of {numeric_cols[0]}",
            "priority": "medium"
        })
    
    # Category comparison
    if categorical_cols and numeric_cols:
        recommendations.append({
            "type": "bar",
            "title": f"Compare {numeric_cols[0]} by {categorical_cols[0]}",
            "description": f"Compare {numeric_cols[0]} across {categorical_cols[0]} categories",
            "priority": "high"
        })
    
    # Correlation analysis
    if len(numeric_cols) >= 2:
        recommendations.append({
            "type": "scatter",
            "title": f"Correlation: {numeric_cols[0]} vs {numeric_cols[1]}",
            "description": f"Explore relationship between {numeric_cols[0]} and {numeric_cols[1]}",
            "priority": "medium"
        })
    
    # Heatmap for multiple numeric columns
    if len(numeric_cols) >= 3:
        recommendations.append({
            "type": "heatmap",
            "title": "Correlation Matrix",
            "description": "Visualize correlations between all numeric variables",
            "priority": "low"
        })
    
    return recommendations[:5]  # Limit to 5 recommendations 