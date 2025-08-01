import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os
from dotenv import load_dotenv

# Import our custom modules
from data_processor import DataProcessor
from llm_analyzer import LLMAnalyzer
from chart_generator import ChartGenerator
from utils import (
    validate_data, format_number, get_chart_insights, 
    create_chart_summary, validate_chart_spec, format_chart_spec_for_display,
    get_chart_recommendations, load_sample_data, export_chart_data
)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Natural Language Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
    }
    .chart-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'llm_analyzer' not in st.session_state:
    st.session_state.llm_analyzer = LLMAnalyzer()
if 'chart_generator' not in st.session_state:
    st.session_state.chart_generator = ChartGenerator()
if 'current_chart' not in st.session_state:
    st.session_state.current_chart = None
if 'chart_history' not in st.session_state:
    st.session_state.chart_history = []
if 'sample_data_loaded' not in st.session_state:
    st.session_state.sample_data_loaded = False

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Natural Language Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("<h4 align='center'>Upload your data and describe what you want to see in plain English!</h4>", unsafe_allow_html=True)
    
    # Help section
    with st.expander("❓ How to Use", expanded=False):
        st.markdown("""
        1. **Upload Data**: Choose a CSV or Excel file
        2. **Describe**: Tell us what you want to see
        3. **Generate**: Click the button to create your chart
        4. **Explore**: Try different prompts and chart types
        
        **Example Prompts:**
        • "Show me a bar chart of sales by region"
        • "Create a line chart showing revenue trends"
        • "Display a pie chart of product categories"
        • "Show correlation between price and sales"
        """)

    # API Key check
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        st.error("⚠️ Gemini API key not found!")
        st.info("Please set your GEMINI_API_KEY in the .env file")
        st.stop()
    else:
        st.success("✅ Gemini API key configured")
    
    # Model settings (hidden but functional)
    model = "gemini-2.0-flash-001"
    temperature = 0.7
    st.session_state.llm_analyzer.model = model
    st.session_state.llm_analyzer.temperature = temperature
    
    # Quick start section
    
    # Responsive layout
    col_quick1, col_or, col_quick2 = st.columns([1, 0.1, 1])
    
    with col_quick1:
        st.subheader("🚀 Quick Start")
        if st.button("📁 Load Sample Data", use_container_width=True):
            sample_data = load_sample_data()
            st.session_state.data_processor.data = sample_data
            st.session_state.data_processor._analyze_data()
            st.session_state.sample_data_loaded = True
            st.success("Sample data loaded!")
            st.rerun()
    
    with col_or:
        st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; height: 100%; justify-content: center;">
            <div style="border-left: 2px solid #ccc; height: 60px; margin: 10px 0; display: block;" class="desktop-line"></div>
            <div style="font-weight: bold; color: #666; margin: 5px 0; font-size: 18px;">OR</div>
            <div style="border-left: 2px solid #ccc; height: 60px; margin: 10px 0; display: block;" class="desktop-line"></div>
        </div>
        <style>
        @media (max-width: 768px) {
            .desktop-line {
                display: none !important;
            }
        }
        </style>
        """, unsafe_allow_html=True)
    
    with col_quick2:
        # File upload section
        st.subheader("📁 Upload Your Data")
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your structured data file"
        )
    
    # Mobile responsive separator
    st.markdown("""
    <style>
    @media (max-width: 768px) {
        .mobile-separator {
            display: block !important;
            text-align: center;
            margin: 20px 0;
        }
        .mobile-separator::before {
            content: '';
            display: inline-block;
            width: 50px;
            height: 2px;
            background-color: #ccc;
            margin: 0 10px;
            vertical-align: middle;
        }
        .mobile-separator::after {
            content: '';
            display: inline-block;
            width: 50px;
            height: 2px;
            background-color: #ccc;
            margin: 0 10px;
            vertical-align: middle;
        }
    }
    @media (min-width: 769px) {
        .mobile-separator {
            display: none !important;
        }
    }
    </style>
    <div class="mobile-separator" style="display: none;">
        <span style="font-weight: bold; color: #666;">OR</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        
        # Check if we have data (either uploaded or sample)
        has_data = False
        data = None
        data_info = None
        data_summary = None
        
        if uploaded_file is not None:
            # Load uploaded data
            if st.session_state.data_processor.load_data(uploaded_file):
                data = st.session_state.data_processor.get_data()
                data_info = st.session_state.data_processor.get_data_info()
                has_data = True
                
                # Data validation
                is_valid, validation_message = validate_data(data)
                if not is_valid:
                    st.error(f"❌ Data validation failed: {validation_message}")
                    return
                
                st.success(f"✅ Data loaded successfully! Shape: {data.shape}")
        elif st.session_state.sample_data_loaded:
            # Use sample data
            data = st.session_state.data_processor.get_data()
            data_info = st.session_state.data_processor.get_data_info()
            has_data = True
            
            st.success(f"✅ Sample data loaded! Shape: {data.shape}")
        
        if has_data:
            # Show data preview
            with st.expander("📋 Data Preview", expanded=False):
                st.dataframe(data.head(10), use_container_width=True)
            
            # Data summary
            data_summary = st.session_state.data_processor.get_data_summary()
            
            # Chart generation section
            st.subheader("🎯 Generate Charts")
            
            # User prompt input
            user_prompt = st.text_area(
                "Describe what chart you want to see:",
                placeholder="e.g., 'Show me a bar chart of sales by region' or 'Create a line chart showing revenue trends over time'",
                height=100,
                help="Describe your visualization needs in natural language"
            )
            
            col_prompt1, col_prompt2 = st.columns([3, 1])
            with col_prompt1:
                if st.button("🚀 Generate Chart", type="primary", use_container_width=True):
                    if user_prompt.strip():
                        with st.spinner("🤖 Analyzing your request..."):
                            # Analyze prompt with LLM
                            chart_spec = st.session_state.llm_analyzer.analyze_prompt(user_prompt, data_info)
                            
                            # Validate chart specification
                            is_valid_spec, spec_message = validate_chart_spec(chart_spec, data_info)
                            if not is_valid_spec:
                                st.error(f"❌ Chart specification error: {spec_message}")
                                return
                            
                            # Generate chart
                            try:
                                chart_fig = st.session_state.chart_generator.create_chart(data, chart_spec)
                                # Check if the chart is empty by inspecting annotations
                                is_empty_chart = False
                                if hasattr(chart_fig, 'layout') and hasattr(chart_fig.layout, 'annotations'):
                                    for ann in chart_fig.layout.annotations or []:
                                        if 'No data available for this chart' in ann.text:
                                            is_empty_chart = True
                                            break
                                if is_empty_chart:
                                    st.error("❌ Unable to generate the requested chart.\n\n**Suggestions:**\n- Check that your prompt specifies valid column names present in your data.\n- Try a different chart type or prompt.\n- Ensure your data has enough numeric or categorical columns for the chosen chart type.\n- For violin/box/histogram, make sure you have at least one numeric column.\n- For bar/line, ensure you have both categorical and numeric columns.")
                                    return
                                st.session_state.current_chart = (chart_spec, chart_fig)
                                # Add to history
                                chart_summary = create_chart_summary(chart_spec, data_summary)
                                st.session_state.chart_history.append({
                                    'timestamp': datetime.now(),
                                    'prompt': user_prompt,
                                    'spec': chart_spec,
                                    'summary': chart_summary
                                })
                                st.success("✅ Chart generated successfully!")
                            except Exception as e:
                                st.error(f"❌ An error occurred while generating the chart: {str(e)}\n\n**Suggestions:**\n- Check your prompt for typos or invalid column names.\n- Try a different chart type.\n- Ensure your data is not empty and has the required columns.")
                                return
                    else:
                        st.warning("Please enter a description of the chart you want to see.")
            
            with col_prompt2:
                if st.button("🎲 Random Chart", use_container_width=True):
                    # Generate a random chart suggestion
                    recommendations = get_chart_recommendations(data_info)
                    if recommendations:
                        import random
                        recommendation = random.choice(recommendations)
                        st.info(f"💡 Try: {recommendation['description']}")
            
            # Display current chart
            if st.session_state.current_chart:
                chart_spec, chart_fig = st.session_state.current_chart
                
                st.subheader("📊 Generated Chart")
                
                # Chart specification
                with st.expander("🔍 Chart Details", expanded=False):
                    st.markdown(format_chart_spec_for_display(chart_spec))
                
                # Display chart
                st.plotly_chart(chart_fig, use_container_width=True)
                
                # Chart insights
                insights = get_chart_insights(data, chart_spec)
                if insights:
                    st.subheader("💡 Insights")
                    for insight in insights:
                        st.markdown(f"• {insight}")
                
                # Export options
                col_export1, col_export2 = st.columns(2)
                with col_export1:
                    if st.button("📥 Download Chart (PNG)"):
                        # Note: In a real implementation, you'd use the download functionality
                        st.info("Chart download functionality would be implemented here")
                
                with col_export2:
                    if st.button("📊 Export Data (CSV)"):
                        csv_data = export_chart_data(data, chart_spec)
                        st.download_button(
                            label="Download CSV",
                            data=csv_data,
                            file_name=f"chart_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
            
            # Chart history
            if st.session_state.chart_history:
                st.subheader("📚 Chart History")
                for i, history_item in enumerate(reversed(st.session_state.chart_history[-5:])):  # Show last 5
                    with st.expander(f"📊 {history_item['summary']['title']} - {history_item['timestamp'].strftime('%H:%M:%S')}"):
                        st.markdown(f"**Prompt:** {history_item['prompt']}")
                        st.markdown(f"**Chart Type:** {history_item['summary']['chart_type']}")
                        if st.button(f"🔄 Regenerate", key=f"regenerate_{i}"):
                            # Regenerate the chart
                            chart_fig = st.session_state.chart_generator.create_chart(data, history_item['spec'])
                            st.session_state.current_chart = (history_item['spec'], chart_fig)
                            st.rerun()
    
    with col2:
        # Data overview
        if has_data and data_info is not None:
            st.subheader("📈 Data Overview")
            
            # Basic metrics
            st.metric("Total Rows", f"{data_summary['total_rows']:,}")
            st.metric("Total Columns", data_summary['total_columns'])
            
            # Column types
            st.markdown("**Column Types:**")
            numeric_count = len(data_info['numeric_columns'])
            categorical_count = len(data_info['categorical_columns'])
            datetime_count = len(data_info['datetime_columns'])
            
            st.markdown(f"• Numeric: {numeric_count}")
            st.markdown(f"• Categorical: {categorical_count}")
            st.markdown(f"• DateTime: {datetime_count}")
            
            # Column suggestions
            suggestions = st.session_state.data_processor.get_column_suggestions()
            
            st.markdown("**Suggested Columns:**")
            if suggestions['x_axis']:
                st.markdown(f"**X-Axis:** {', '.join(suggestions['x_axis'][:3])}")
            if suggestions['y_axis']:
                st.markdown(f"**Y-Axis:** {', '.join(suggestions['y_axis'][:3])}")
            if suggestions['color']:
                st.markdown(f"**Color:** {', '.join(suggestions['color'][:3])}")
            
            # Chart recommendations
            recommendations = get_chart_recommendations(data_info)
            if recommendations:
                st.subheader("💡 Recommended Charts")
                for i, rec in enumerate(recommendations[:3]):
                    priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}[rec['priority']]
                    st.markdown(f"{priority_color} **{rec['title']}**")
                    st.markdown(f"*{rec['description']}*")
                    if i < len(recommendations) - 1:
                        st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built by Subham Moda using Streamlit, Plotly, and Google Gemini | © SubhamModa 2025"
    )

if __name__ == "__main__":
    main() 