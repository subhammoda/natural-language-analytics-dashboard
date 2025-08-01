import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional, Tuple
import streamlit as st

class ChartGenerator:
    """Generates various types of charts using Plotly based on specifications."""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
        self.default_height = 600
        self.default_width = 800
    
    def create_chart(self, data: pd.DataFrame, chart_spec: Dict) -> go.Figure:
        """
        Create a chart based on the specification.
        
        Args:
            data: Input DataFrame
            chart_spec: Chart specification from LLM
            
        Returns:
            go.Figure: Plotly figure object
        """
        # Clean up chart specification to remove invalid column references
        chart_spec = self._clean_chart_spec(chart_spec, data)
        
        chart_type = chart_spec.get('chart_type', 'bar').lower()
        
        # Prepare data based on specification
        processed_data = self._prepare_data(data, chart_spec)
        
        # Create chart based on type
        if chart_type == 'bar':
            return self._create_bar_chart(processed_data, chart_spec)
        elif chart_type == 'line':
            return self._create_line_chart(processed_data, chart_spec)
        elif chart_type == 'scatter':
            return self._create_scatter_chart(processed_data, chart_spec)
        elif chart_type == 'pie':
            return self._create_pie_chart(processed_data, chart_spec)
        elif chart_type == 'histogram':
            return self._create_histogram_chart(processed_data, chart_spec)
        elif chart_type == 'box':
            return self._create_box_chart(processed_data, chart_spec)
        elif chart_type == 'heatmap':
            return self._create_heatmap_chart(processed_data, chart_spec)
        elif chart_type == 'area':
            return self._create_area_chart(processed_data, chart_spec)
        elif chart_type == 'bubble':
            return self._create_bubble_chart(processed_data, chart_spec)
        elif chart_type == 'violin':
            return self._create_violin_chart(processed_data, chart_spec)
        else:
            # Default to bar chart
            return self._create_bar_chart(processed_data, chart_spec)
    
    def _prepare_data(self, data: pd.DataFrame, chart_spec: Dict) -> pd.DataFrame:
        """Prepare and aggregate data based on chart specification."""
        if data.empty:
            return data
        
        x_axis = chart_spec.get('x_axis')
        y_axis = chart_spec.get('y_axis')
        color = chart_spec.get('color')
        aggregation = chart_spec.get('aggregation', 'sum')
        limit = chart_spec.get('limit')
        sort_by = chart_spec.get('sort_by')
        
        # If both x and y axes are specified, aggregate the data
        if x_axis and y_axis and x_axis in data.columns and y_axis in data.columns:
            # Check if we need to preserve color column for grouping
            if color and color in data.columns:
                # Group by both x_axis and color to preserve color information
                group_cols = [x_axis, color]
                if aggregation == 'count':
                    processed_data = data.groupby(group_cols).size().reset_index(name=y_axis)
                else:
                    agg_func = getattr(data.groupby(group_cols)[y_axis], aggregation)
                    processed_data = agg_func().reset_index()
            else:
                # Standard aggregation without color grouping
                if aggregation == 'count':
                    processed_data = data.groupby(x_axis).size().reset_index(name=y_axis)
                else:
                    agg_func = getattr(data.groupby(x_axis)[y_axis], aggregation)
                    processed_data = agg_func().reset_index()
        # Ensure processed_data is defined before further use
        if 'processed_data' not in locals():
            processed_data = data.copy()
        
        # Convert date columns to datetime if they exist
        date_columns = []
        for col in processed_data.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    # Try to parse with specific format first (for DD-MMM-YY format)
                    if processed_data[col].dtype == 'object':
                        # Check if it's in DD-MMM-YY format (like "04-Jan-22")
                        sample_date = str(processed_data[col].iloc[0])
                        if len(sample_date.split('-')) == 3 and len(sample_date.split('-')[2]) == 2:
                            processed_data[col] = pd.to_datetime(processed_data[col], format='%d-%b-%y')
                        else:
                            processed_data[col] = pd.to_datetime(processed_data[col])
                    else:
                        processed_data[col] = pd.to_datetime(processed_data[col])
                    
                    # Convert to string format to avoid deprecation warning
                    processed_data[col] = processed_data[col].dt.strftime('%Y-%m-%d')
                    date_columns.append(col)
                except:
                    pass  # Skip if conversion fails
        
        # Sort if specified
        if sort_by and sort_by in processed_data.columns:
            processed_data = processed_data.sort_values(sort_by, ascending=False)
        elif y_axis in processed_data.columns:
            processed_data = processed_data.sort_values(y_axis, ascending=False)
        
        # Auto-sort date columns in ascending order (now as strings)
        date_cols = [col for col in processed_data.columns if col in date_columns]
        if len(date_cols) > 0:
            # Sort by the first date column in ascending order
            date_col = date_cols[0]
            processed_data = processed_data.sort_values(date_col, ascending=True)
        
        # Limit results if specified
        if limit and limit > 0:
            processed_data = processed_data.head(limit)
        
        return processed_data
    
    def _create_bar_chart(self, data: pd.DataFrame, chart_spec: Dict) -> go.Figure:
        """Create a bar chart."""
        x_axis = chart_spec.get('x_axis')
        y_axis = chart_spec.get('y_axis')
        color = chart_spec.get('color')
        title = chart_spec.get('title', 'Bar Chart')
        orientation = chart_spec.get('orientation', 'vertical')
        
        if data.empty or (x_axis and x_axis not in data.columns):
            return self._create_empty_chart(title)
        
        # Determine x and y columns
        if x_axis and y_axis and x_axis in data.columns and y_axis in data.columns:
            x_col, y_col = x_axis, y_axis
        elif x_axis and x_axis in data.columns:
            x_col, y_col = x_axis, data.columns[0]
        else:
            x_col, y_col = data.columns[0], data.columns[1] if len(data.columns) > 1 else data.columns[0]
        
        # Validate color column exists in data
        color_col = color if color and color in data.columns else None
        
        # Create the chart
        if orientation == 'horizontal':
            fig = px.bar(data, y=x_col, x=y_col, color=color_col, title=title)
        else:
            fig = px.bar(data, x=x_col, y=y_col, color=color_col, title=title)
        
        fig.update_layout(
            height=self.default_height,
            width=self.default_width,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def _create_line_chart(self, data: pd.DataFrame, chart_spec: Dict) -> go.Figure:
        """Create a line chart."""
        x_axis = chart_spec.get('x_axis')
        y_axis = chart_spec.get('y_axis')
        color = chart_spec.get('color')
        title = chart_spec.get('title', 'Line Chart')
        
        if data.empty:
            return self._create_empty_chart(title)
        
        # Determine x and y columns
        if x_axis and y_axis and x_axis in data.columns and y_axis in data.columns:
            x_col, y_col = x_axis, y_axis
        elif x_axis and x_axis in data.columns:
            x_col, y_col = x_axis, data.columns[0]
        else:
            x_col, y_col = data.columns[0], data.columns[1] if len(data.columns) > 1 else data.columns[0]
        
        # Validate color column exists in data
        color_col = color if color and color in data.columns else None
        
        fig = px.line(data, x=x_col, y=y_col, color=color_col, title=title)
        
        fig.update_layout(
            height=self.default_height,
            width=self.default_width,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def _create_scatter_chart(self, data: pd.DataFrame, chart_spec: Dict) -> go.Figure:
        """Create a scatter chart."""
        x_axis = chart_spec.get('x_axis')
        y_axis = chart_spec.get('y_axis')
        color = chart_spec.get('color')
        size = chart_spec.get('size')
        title = chart_spec.get('title', 'Scatter Plot')
        
        if data.empty:
            return self._create_empty_chart(title)
        
        # Determine x and y columns
        if x_axis and y_axis and x_axis in data.columns and y_axis in data.columns:
            x_col, y_col = x_axis, y_axis
        else:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                x_col, y_col = numeric_cols[0], numeric_cols[1]
            else:
                return self._create_empty_chart(title)
        
        # Validate color and size columns exist in data
        color_col = color if color and color in data.columns else None
        size_col = size if size and size in data.columns else None
        
        fig = px.scatter(data, x=x_col, y=y_col, color=color_col, size=size_col, title=title)
        
        fig.update_layout(
            height=self.default_height,
            width=self.default_width,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def _create_pie_chart(self, data: pd.DataFrame, chart_spec: Dict) -> go.Figure:
        """Create a pie chart."""
        x_axis = chart_spec.get('x_axis')
        y_axis = chart_spec.get('y_axis')
        title = chart_spec.get('title', 'Pie Chart')
        
        if data.empty:
            return self._create_empty_chart(title)
        
        # Determine columns
        if x_axis and y_axis and x_axis in data.columns and y_axis in data.columns:
            names_col, values_col = x_axis, y_axis
        elif x_axis and x_axis in data.columns:
            names_col = x_axis
            values_col = data.columns[0] if data.columns[0] != x_axis else data.columns[1] if len(data.columns) > 1 else x_axis
        else:
            names_col = data.columns[0]
            values_col = data.columns[1] if len(data.columns) > 1 else data.columns[0]
        
        fig = px.pie(data, names=names_col, values=values_col, title=title)
        
        fig.update_layout(
            height=self.default_height,
            width=self.default_width,
            template='plotly_white'
        )
        
        return fig
    
    def _create_histogram_chart(self, data: pd.DataFrame, chart_spec: Dict) -> go.Figure:
        """Create a histogram chart."""
        x_axis = chart_spec.get('x_axis')
        color = chart_spec.get('color')
        title = chart_spec.get('title', 'Histogram')
        
        if data.empty:
            return self._create_empty_chart(title)
        
        # Determine x column
        if x_axis and x_axis in data.columns:
            x_col = x_axis
        else:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                x_col = numeric_cols[0]
            else:
                return self._create_empty_chart(title)
        
        # Validate color column exists in data
        color_col = color if color and color in data.columns else None
        
        fig = px.histogram(data, x=x_col, color=color_col, title=title)
        
        fig.update_layout(
            height=self.default_height,
            width=self.default_width,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def _create_box_chart(self, data: pd.DataFrame, chart_spec: Dict) -> go.Figure:
        """Create a box chart."""
        x_axis = chart_spec.get('x_axis')
        y_axis = chart_spec.get('y_axis')
        color = chart_spec.get('color')
        title = chart_spec.get('title', 'Box Plot')
        
        if data.empty:
            return self._create_empty_chart(title)
        
        # Determine x and y columns
        if x_axis and y_axis and x_axis in data.columns and y_axis in data.columns:
            x_col, y_col = x_axis, y_axis
        else:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            categorical_cols = data.select_dtypes(include=['object']).columns
            
            if len(numeric_cols) > 0 and len(categorical_cols) > 0:
                x_col, y_col = categorical_cols[0], numeric_cols[0]
            elif len(numeric_cols) > 0:
                x_col, y_col = None, numeric_cols[0]
            else:
                return self._create_empty_chart(title)
        
        # Validate color column exists in data
        color_col = color if color and color in data.columns else None
        
        if x_col:
            fig = px.box(data, x=x_col, y=y_col, color=color_col, title=title)
        else:
            fig = px.box(data, y=y_col, color=color_col, title=title)
        
        fig.update_layout(
            height=self.default_height,
            width=self.default_width,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def _create_heatmap_chart(self, data: pd.DataFrame, chart_spec: Dict) -> go.Figure:
        """Create a heatmap chart."""
        x_axis = chart_spec.get('x_axis')
        y_axis = chart_spec.get('y_axis')
        title = chart_spec.get('title', 'Heatmap')
        
        if data.empty:
            return self._create_empty_chart(title)
        
        # For correlation heatmap
        if x_axis is None and y_axis is None:
            numeric_data = data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) < 2:
                return self._create_empty_chart(title)
            
            corr_matrix = numeric_data.corr()
            fig = px.imshow(
                corr_matrix,
                title=title,
                color_continuous_scale='RdBu',
                aspect='auto'
            )
        else:
            # Pivot table heatmap
            if x_axis and y_axis and x_axis in data.columns and y_axis in data.columns:
                pivot_data = data.pivot_table(
                    index=y_axis,
                    columns=x_axis,
                    aggfunc='size',
                    fill_value=0
                )
                fig = px.imshow(
                    pivot_data,
                    title=title,
                    color_continuous_scale='Viridis',
                    aspect='auto'
                )
            else:
                return self._create_empty_chart(title)
        
        fig.update_layout(
            height=self.default_height,
            width=self.default_width,
            template='plotly_white'
        )
        
        return fig
    
    def _create_area_chart(self, data: pd.DataFrame, chart_spec: Dict) -> go.Figure:
        """Create an area chart."""
        x_axis = chart_spec.get('x_axis')
        y_axis = chart_spec.get('y_axis')
        color = chart_spec.get('color')
        title = chart_spec.get('title', 'Area Chart')
        
        if data.empty:
            return self._create_empty_chart(title)
        
        # Determine x and y columns
        if x_axis and y_axis and x_axis in data.columns and y_axis in data.columns:
            x_col, y_col = x_axis, y_axis
        elif x_axis and x_axis in data.columns:
            x_col, y_col = x_axis, data.columns[0]
        else:
            x_col, y_col = data.columns[0], data.columns[1] if len(data.columns) > 1 else data.columns[0]
        
        # Validate color column exists in data
        color_col = color if color and color in data.columns else None
        
        fig = px.area(data, x=x_col, y=y_col, color=color_col, title=title)
        
        fig.update_layout(
            height=self.default_height,
            width=self.default_width,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def _create_bubble_chart(self, data: pd.DataFrame, chart_spec: Dict) -> go.Figure:
        """Create a bubble chart."""
        x_axis = chart_spec.get('x_axis')
        y_axis = chart_spec.get('y_axis')
        size = chart_spec.get('size')
        color = chart_spec.get('color')
        title = chart_spec.get('title', 'Bubble Chart')
        
        if data.empty:
            return self._create_empty_chart(title)
        
        # Determine columns
        if x_axis and y_axis and x_axis in data.columns and y_axis in data.columns:
            x_col, y_col = x_axis, y_axis
        else:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                x_col, y_col = numeric_cols[0], numeric_cols[1]
            else:
                return self._create_empty_chart(title)
        
        # Validate color and size columns exist in data
        color_col = color if color and color in data.columns else None
        size_col = size if size and size in data.columns else None
        
        fig = px.scatter(data, x=x_col, y=y_col, size=size_col, color=color_col, title=title)
        
        fig.update_layout(
            height=self.default_height,
            width=self.default_width,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def _create_violin_chart(self, data: pd.DataFrame, chart_spec: Dict) -> go.Figure:
        """Create a violin chart."""
        x_axis = chart_spec.get('x_axis')
        y_axis = chart_spec.get('y_axis')
        color = chart_spec.get('color')
        title = chart_spec.get('title', 'Violin Plot')
        
        if data.empty:
            return self._create_empty_chart(title)
        
        # Determine x and y columns
        if x_axis and y_axis and x_axis in data.columns and y_axis in data.columns:
            x_col, y_col = x_axis, y_axis
        else:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            categorical_cols = data.select_dtypes(include=['object']).columns
            
            if len(numeric_cols) > 0 and len(categorical_cols) > 0:
                x_col, y_col = categorical_cols[0], numeric_cols[0]
            elif len(numeric_cols) > 0:
                x_col, y_col = None, numeric_cols[0]
            else:
                return self._create_empty_chart(title)
        
        # Validate color column exists in data
        color_col = color if color and color in data.columns else None
        
        if x_col:
            fig = px.violin(data, x=x_col, y=y_col, color=color_col, title=title)
        else:
            fig = px.violin(data, y=y_col, color=color_col, title=title)
        
        fig.update_layout(
            height=self.default_height,
            width=self.default_width,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def _clean_chart_spec(self, chart_spec: Dict, data: pd.DataFrame) -> Dict:
        """Clean up chart specification by removing invalid column references."""
        cleaned_spec = chart_spec.copy()
        available_columns = data.columns.tolist()
        
        # Check and remove invalid column references
        column_fields = ['x_axis', 'y_axis', 'color', 'size', 'facet', 'sort_by']
        
        for field in column_fields:
            if field in cleaned_spec and cleaned_spec[field]:
                if cleaned_spec[field] not in available_columns:
                    # Remove invalid column reference
                    cleaned_spec[field] = None
                    print(f"Warning: Column '{cleaned_spec[field]}' not found in data. Removing from {field}.")
        
        return cleaned_spec
    
    def _create_empty_chart(self, title: str) -> go.Figure:
        """Create an empty chart when data is not available."""
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for this chart",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title=title,
            height=self.default_height,
            width=self.default_width,
            template='plotly_white'
        )
        return fig
    
    def create_dashboard(self, charts: List[Tuple[str, go.Figure]]) -> go.Figure:
        """Create a dashboard with multiple charts."""
        if not charts:
            return self._create_empty_chart("Dashboard")
        
        n_charts = len(charts)
        cols = min(2, n_charts)
        rows = (n_charts + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[title for title, _ in charts],
            specs=[[{"secondary_y": False}] * cols] * rows
        )
        
        for i, (title, chart) in enumerate(charts):
            row = i // cols + 1
            col = i % cols + 1
            
            for trace in chart.data:
                fig.add_trace(trace, row=row, col=col)
        
        fig.update_layout(
            height=300 * rows,
            width=400 * cols,
            showlegend=False,
            template='plotly_white'
        )
        
        return fig 