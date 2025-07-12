import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import streamlit as st
import io

class DataProcessor:
    """Handles data loading, preprocessing, and analysis for CSV/Excel files."""
    
    def __init__(self):
        self.data = None
        self.data_info = {}
    
    def load_data(self, uploaded_file) -> bool:
        """
        Load data from uploaded file (CSV or Excel).
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if uploaded_file is None:
                return False
                
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                self.data = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                self.data = pd.read_excel(uploaded_file)
            else:
                st.error(f"Unsupported file format: {file_extension}")
                return False
            
            self._analyze_data()
            return True
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return False
    
    def _analyze_data(self):
        """Analyze the loaded data and store metadata."""
        if self.data is None:
            return
            
        self.data_info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'numeric_columns': list(self.data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.data.select_dtypes(include=['object']).columns),
            'datetime_columns': list(self.data.select_dtypes(include=['datetime64']).columns),
            'sample_data': self.data.head().to_dict('records')
        }
    
    def get_data_summary(self) -> Dict:
        """Get a comprehensive summary of the data."""
        if self.data is None:
            return {}
            
        summary = {
            'total_rows': len(self.data),
            'total_columns': len(self.data.columns),
            'column_info': {}
        }
        
        for col in self.data.columns:
            col_info = {
                'dtype': str(self.data[col].dtype),
                'missing_count': self.data[col].isnull().sum(),
                'missing_percentage': round(self.data[col].isnull().sum() / len(self.data) * 100, 2)
            }
            
            if self.data[col].dtype in ['int64', 'float64']:
                col_info.update({
                    'min': float(self.data[col].min()) if not self.data[col].isnull().all() else None,
                    'max': float(self.data[col].max()) if not self.data[col].isnull().all() else None,
                    'mean': float(self.data[col].mean()) if not self.data[col].isnull().all() else None,
                    'std': float(self.data[col].std()) if not self.data[col].isnull().all() else None
                })
            elif self.data[col].dtype == 'object':
                col_info.update({
                    'unique_count': self.data[col].nunique(),
                    'most_common': self.data[col].mode().iloc[0] if not self.data[col].mode().empty else None
                })
            
            summary['column_info'][col] = col_info
        
        return summary
    
    def get_column_suggestions(self) -> Dict[str, List[str]]:
        """Get suggestions for different types of analysis based on column types."""
        suggestions = {
            'x_axis': [],
            'y_axis': [],
            'color': [],
            'size': [],
            'facet': []
        }
        
        if self.data is None:
            return suggestions
        
        # X-axis suggestions (categorical or datetime)
        suggestions['x_axis'] = self.data_info['categorical_columns'] + self.data_info['datetime_columns']
        
        # Y-axis suggestions (numeric)
        suggestions['y_axis'] = self.data_info['numeric_columns']
        
        # Color suggestions (categorical)
        suggestions['color'] = self.data_info['categorical_columns']
        
        # Size suggestions (numeric)
        suggestions['size'] = self.data_info['numeric_columns']
        
        # Facet suggestions (categorical with limited unique values)
        for col in self.data_info['categorical_columns']:
            if self.data[col].nunique() <= 10:  # Only suggest for columns with <= 10 unique values
                suggestions['facet'].append(col)
        
        return suggestions
    
    def get_data(self) -> Optional[pd.DataFrame]:
        """Get the loaded data."""
        return self.data
    
    def get_data_info(self) -> Dict:
        """Get the data analysis information."""
        return self.data_info
    
    def filter_data(self, filters: Dict) -> pd.DataFrame:
        """
        Apply filters to the data.
        
        Args:
            filters: Dictionary of column: value pairs to filter by
            
        Returns:
            pd.DataFrame: Filtered data
        """
        if self.data is None:
            return pd.DataFrame()
        
        filtered_data = self.data.copy()
        
        for column, value in filters.items():
            if column in filtered_data.columns and value is not None:
                if isinstance(value, (list, tuple)):
                    filtered_data = filtered_data[filtered_data[column].isin(value)]
                else:
                    filtered_data = filtered_data[filtered_data[column] == value]
        
        return filtered_data
    
    def get_unique_values(self, column: str) -> List:
        """Get unique values for a specific column."""
        if self.data is None or column not in self.data.columns:
            return []
        
        return sorted(self.data[column].dropna().unique().tolist())
    
    def get_column_statistics(self, column: str) -> Dict:
        """Get detailed statistics for a specific column."""
        if self.data is None or column not in self.data.columns:
            return {}
        
        col_data = self.data[column].dropna()
        
        stats = {
            'count': len(col_data),
            'missing_count': self.data[column].isnull().sum(),
            'dtype': str(self.data[column].dtype)
        }
        
        if self.data[column].dtype in ['int64', 'float64']:
            stats.update({
                'min': float(col_data.min()) if len(col_data) > 0 else None,
                'max': float(col_data.max()) if len(col_data) > 0 else None,
                'mean': float(col_data.mean()) if len(col_data) > 0 else None,
                'median': float(col_data.median()) if len(col_data) > 0 else None,
                'std': float(col_data.std()) if len(col_data) > 0 else None,
                'q25': float(col_data.quantile(0.25)) if len(col_data) > 0 else None,
                'q75': float(col_data.quantile(0.75)) if len(col_data) > 0 else None
            })
        elif self.data[column].dtype == 'object':
            stats.update({
                'unique_count': col_data.nunique(),
                'most_common': col_data.mode().iloc[0] if not col_data.mode().empty else None,
                'most_common_count': col_data.value_counts().iloc[0] if len(col_data) > 0 else 0
            })
        
        return stats 