import streamlit as st
import requests
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Tweet Analysis Platform",
    page_icon="🐦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
def load_css():
    css_file = Path("./frontend/styling/style.css")
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("style.css file not found. Please ensure it's in the same directory.")

load_css()

# Header
st.markdown("""
    <div class="header-container">
        <h1 class="main-title">🐦 Tweet Analysis Platform</h1>
        <p class="subtitle">Analyze and categorize tweets with AI-powered insights</p>
    </div>
""", unsafe_allow_html=True)

# Main form container
with st.container():
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    
    # Tweet text input
    st.markdown('<label class="input-label">Tweet Text</label>', unsafe_allow_html=True)
    tweet_text = st.text_area(
        "tweet_text",
        placeholder="Enter your tweet text here...",
        height=150,
        label_visibility="collapsed",
        key="tweet_input"
    )
    
    # Create two columns for dropdowns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<label class="input-label">Topic</label>', unsafe_allow_html=True)
        topic = st.selectbox(
            "topic",
            options=[
                "Technology",
                "Politics",
                "Sports",
                "Entertainment",
                "Business",
                "Science",
                "Health",
                "Education",
                "Travel",
                "Food",
                "Fashion",
                "Other"
            ],
            label_visibility="collapsed",
            key="topic_select"
        )
        
        st.markdown('<label class="input-label">Language</label>', unsafe_allow_html=True)
        language = st.selectbox(
            "language",
            options=[
                "English",
                "Spanish",
                "French",
                "German",
                "Italian",
                "Portuguese",
                "Dutch",
                "Japanese",
                "Chinese",
                "Arabic",
                "Hindi",
                "Other"
            ],
            label_visibility="collapsed",
            key="language_select"
        )
    
    with col2:
        st.markdown('<label class="input-label">Source</label>', unsafe_allow_html=True)
        source = st.selectbox(
            "source",
            options=[
                "Twitter",
                "Kaggle",
                "Website",
                "API",
                "Manual Entry"
            ],
            label_visibility="collapsed",
            key="source_select"
        )
        
        st.markdown('<label class="input-label">Model</label>', unsafe_allow_html=True)
        model = st.selectbox(
            "model",
            options=[
                "logistic_regression",
                "naive_bayes",
                "random_forest",
                "gradient_boosting",
                "all"
            ],
            format_func=lambda x: x.replace("_", " ").title(),
            label_visibility="collapsed",
            key="model_select"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Submit button
st.markdown('<div class="button-container">', unsafe_allow_html=True)
if st.button("🚀 Analyze Tweet", use_container_width=True, type="primary"):
    if not tweet_text.strip():
        st.error("⚠️ Please enter tweet text before submitting.")
    else:
        with st.spinner("Analyzing your tweet..."):
            try:
                # Prepare data for API
                data = {
                    "texts": [tweet_text],
                    "classifier_type": model
                }
                
                response = requests.post("http://localhost:8000/predict", json=data)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Tweet analyzed successfully!")
                    
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.markdown("### 📊 Analysis Results")
                    
                    # Display metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Topic", topic)
                    with col2:
                        st.metric("Language", language)
                    with col3:
                        st.metric("Source", source)
                    
                    st.markdown('<div style="margin: 1.5rem 0;"></div>', unsafe_allow_html=True)
                    
                    # If "all" models selected, display results for each model
                    if model == "all":
                        st.markdown("### 🤖 Model Predictions")
                        
                        # Assuming the API returns predictions for all models when "all" is selected
                        for prediction in result.get("predictions", []):
                            model_name = result.get("model", "Unknown Model")
                            label = prediction.get("label", "N/A")
                            confidence = prediction.get("confidence", 0)
                            
                            # Determine sentiment color
                            if label.lower() == "positive":
                                sentiment_color = "#2E7D32"
                                emoji = "😊"
                            elif label.lower() == "negative":
                                sentiment_color = "#D32F2F"
                                emoji = "😞"
                            else:
                                sentiment_color = "#F57C00"
                                emoji = "😐"
                            
                            st.markdown(f"""
                                <div class="prediction-card">
                                    <div class="model-badge">{model_name.replace("_", " ").title()}</div>
                                    <div class="prediction-content">
                                        <div class="sentiment-label" style="color: {sentiment_color};">
                                            {emoji} {label}
                                        </div>
                                        <div class="confidence-bar-container">
                                            <div class="confidence-label">Confidence: {confidence:.2%}</div>
                                            <div class="confidence-bar">
                                                <div class="confidence-fill" style="width: {confidence * 100}%; background: linear-gradient(90deg, {sentiment_color}, {sentiment_color}AA);"></div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                    
                    # Single model prediction
                    else:
                        prediction = result.get("predictions", [{}])[0]
                        model_name = result.get("model", model)
                        label = prediction.get("label", "N/A")
                        confidence = prediction.get("confidence", 0)
                        
                        # Determine sentiment color and emoji
                        if label.lower() == "positive":
                            sentiment_color = "#2E7D32"
                            emoji = "😊"
                            bg_gradient = "linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%)"
                        elif label.lower() == "negative":
                            sentiment_color = "#D32F2F"
                            emoji = "😞"
                            bg_gradient = "linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%)"
                        else:
                            sentiment_color = "#F57C00"
                            emoji = "😐"
                            bg_gradient = "linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%)"
                        
                        st.markdown(f"""
                            <div class="single-prediction-card" style="background: {bg_gradient};">
                                <div class="model-name">Model: {model_name.replace("_", " ").title()}</div>
                                <div class="sentiment-result">
                                    <div class="sentiment-emoji">{emoji}</div>
                                    <div class="sentiment-text" style="color: {sentiment_color};">{label}</div>
                                </div>
                                <div class="confidence-section">
                                    <div class="confidence-label-large">Confidence Score</div>
                                    <div class="confidence-value">{confidence:.2%}</div>
                                    <div class="confidence-bar-large">
                                        <div class="confidence-fill-large" style="width: {confidence * 100}%; background: {sentiment_color};"></div>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Display tweet content in expander
                    with st.expander("📝 View Tweet Content"):
                        st.write(tweet_text)
                        st.caption(f"Character count: {len(tweet_text)}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                else:
                    st.error(f"❌ Error: {response.status_code} - {response.text}")
                
            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to the backend. Please ensure the API is running on http://localhost:8000")
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        <p>Built with ❤️ using Streamlit | Powered by AI</p>
    </div>
""", unsafe_allow_html=True)