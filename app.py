import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sentiment_predictor import SentimentPredictor
import sys
sys.path.insert(0, r'c:\Users\Lenovo\Downloads')
from whatsapp_analytics import WhatsAppAnalytics
import re

# Set page theme
st.set_page_config(
    page_title="WhatsApp Sentiment Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to parse WhatsApp TXT files
def parse_whatsapp_txt(txt_content):
    """Parse WhatsApp TXT export to DataFrame"""
    messages = []
    
    patterns = [
        r'^(\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s+[AP]M)?)\s*[-–]\s*(.+?):\s*(.*)$',
        r'^(\d{1,2}-\d{1,2}-\d{2,4}\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s+[AP]M)?)\s*[-–]\s*(.+?):\s*(.*)$',
        r'^\[?(\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}:\d{2})\]?\s*(.+?):\s*(.*)$',
    ]
    
    for line in txt_content.split('\n'):
        line = line.strip()
        
        if not line or any(x in line for x in [' created this group', ' added ', ' left', ' removed ', ' changed']):
            continue
        
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                timestamp, sender, message = match.groups()
                messages.append({
                    'timestamp': timestamp.strip(),
                    'sender': sender.strip(),
                    'message': message.strip()
                })
                break
    
    if not messages:
        raise ValueError("Could not parse any messages. Check format.")
    
    return pd.DataFrame(messages)

def sentiment_color(sentiment):
    """Return color based on sentiment"""
    colors = {'positive': '#00D084', 'neutral': '#FFD700', 'negative': '#FF4444'}
    return colors.get(sentiment, '#808080')

def get_sentiment_emoji(sentiment):
    """Get emoji for sentiment"""
    emojis = {'positive': '😊', 'neutral': '😐', 'negative': '😞'}
    return emojis.get(sentiment, '❓')

# Load models
@st.cache_resource
def load_predictor():
    return SentimentPredictor()

predictor = load_predictor()

# Header
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>💬 WhatsApp Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #666;'>Beautiful Analytics + Sentiment Analysis</h4>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Single Text", "📁 Batch Upload", "📊 Examples", "📱 Chat Analytics", "ℹ️ About"])

# ==================== TAB 1: SINGLE TEXT ====================
with tab1:
    st.header("💬 Single Text Sentiment Analysis")
    st.markdown("Type any text and get instant sentiment analysis!")
    
    text_input = st.text_area("Enter text:", height=120, placeholder="E.g., 'This is amazing!' or 'This is terrible'")
    
    if st.button("🎯 Analyze Sentiment", use_container_width=True):
        if text_input.strip():
            try:
                prediction = predictor.predict(text_input)
                
                sentiment = prediction['sentiment']
                confidence = prediction['confidence']
                
                col_res1, col_res2, col_res3 = st.columns(3)
                
                with col_res1:
                    st.metric("Sentiment", f"{get_sentiment_emoji(sentiment)} {sentiment.upper()}")
                
                with col_res2:
                    st.metric("Confidence", f"{confidence*100:.1f}%")
                
                with col_res3:
                    st.metric("Status", "✓ Analyzed")
                
                # Probability chart
                st.subheader("📊 Probability Distribution")
                probs = prediction['probabilities']
                
                fig = go.Figure(data=[go.Pie(
                    labels=['😊 Positive', '😐 Neutral', '😞 Negative'],
                    values=[probs['positive']*100, probs['neutral']*100, probs['negative']*100],
                    marker=dict(colors=['#00D084', '#FFD700', '#FF4444']),
                    hole=0.3
                )])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Probability bars
                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    st.write(f"😊 **Positive**: {probs['positive']*100:.1f}%")
                with col_p2:
                    st.write(f"😐 **Neutral**: {probs['neutral']*100:.1f}%")
                with col_p3:
                    st.write(f"😞 **Negative**: {probs['negative']*100:.1f}%")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please enter text")

# ==================== TAB 2: BATCH UPLOAD ====================
with tab2:
    st.header("📁 Batch Sentiment Analysis")
    st.markdown("Upload CSV or paste messages to analyze multiple texts!")
    
    col_opt1, col_opt2 = st.columns(2)
    
    with col_opt1:
        st.subheader("📤 Upload CSV")
        uploaded_file = st.file_uploader("CSV with 'message' column", type=['csv'], key='batch_csv')
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(f"📋 Loaded {len(df)} rows")
                
                if 'message' in df.columns and st.button("Analyze CSV", use_container_width=True, key='btn_csv'):
                    with st.spinner("🔄 Analyzing..."):
                        predictions = predictor.predict_batch(df['message'].tolist())
                        results_df = df.copy()
                        results_df['sentiment'] = [p['sentiment'] for p in predictions]
                        results_df['confidence'] = [p['confidence'] for p in predictions]
                        
                        st.dataframe(results_df, use_container_width=True, hide_index=True)
                        
                        # Stats
                        st.subheader("📊 Summary")
                        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                        with col_s1:
                            st.metric("Total", len(results_df))
                        with col_s2:
                            st.metric("😊 Positive", (results_df['sentiment'] == 'positive').sum())
                        with col_s3:
                            st.metric("😐 Neutral", (results_df['sentiment'] == 'neutral').sum())
                        with col_s4:
                            st.metric("😞 Negative", (results_df['sentiment'] == 'negative').sum())
                        
                        # Pie
                        sent_counts = results_df['sentiment'].value_counts()
                        fig = go.Figure(data=[go.Pie(
                            labels=sent_counts.index,
                            values=sent_counts.values,
                            marker=dict(colors=['#00D084' if x == 'positive' else '#FFD700' if x == 'neutral' else '#FF4444' for x in sent_counts.index])
                        )])
                        st.plotly_chart(fig, use_container_width=True)
                        
                        csv = results_df.to_csv(index=False)
                        st.download_button("📥 Download", csv, "results.csv", "text/csv")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    with col_opt2:
        st.subheader("📝 Paste Messages")
        text_list = st.text_area("One message per line:", height=300)
        
        if st.button("Analyze Text", use_container_width=True, key='btn_text'):
            if text_list.strip():
                messages = [m.strip() for m in text_list.split('\n') if m.strip()]
                with st.spinner("🔄 Analyzing..."):
                    predictions = predictor.predict_batch(messages)
                    results = []
                    for msg, pred in zip(messages, predictions):
                        results.append({'message': msg, 'sentiment': pred['sentiment'], 'confidence': pred['confidence']})
                    
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    fig = go.Figure(data=[go.Bar(
                        x=results_df['sentiment'].value_counts().index,
                        y=results_df['sentiment'].value_counts().values,
                        marker_color=['#00D084', '#FFD700', '#FF4444'][:len(results_df['sentiment'].value_counts())]
                    )])
                    st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 3: EXAMPLES ====================
with tab3:
    st.header("📊 Fun Examples")
    st.markdown("See how the analyzer works on real examples!")
    
    examples = [
        "🎉 I absolutely love this! Amazing experience!",
        "😤 This is terrible and I'm very disappointed.",
        "😐 The weather is nice today.",
        "😡 You're the worst person ever!",
        "😊 It's okay, nothing special.",
        "❤️ I can't imagine my life without you!",
        "😩 This is exhausting and frustrating.",
        "😍 This is absolutely incredible!",
    ]
    
    col_e1, col_e2 = st.columns(2)
    
    for idx, example in enumerate(examples):
        col = col_e1 if idx % 2 == 0 else col_e2
        
        with col:
            pred = predictor.predict(example)
            sent = pred['sentiment']
            color = sentiment_color(sent)
            emoji = get_sentiment_emoji(sent)
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {color}22 0%, {color}11 100%); 
                        border-left: 4px solid {color}; padding: 15px; border-radius: 8px; margin: 10px 0;'>
            <p style='margin: 0; font-style: italic; color: #333;'>{example}</p>
            <p style='margin: 8px 0 0 0; font-weight: bold; color: {color};'>{emoji} {sent.upper()} ({pred['confidence']*100:.0f}%)</p>
            </div>
            """, unsafe_allow_html=True)

# ==================== TAB 4: CHAT ANALYTICS ====================
with tab4:
    st.header("📱 WhatsApp Chat Analytics")
    
    uploaded_chat = st.file_uploader("Upload WhatsApp export (CSV or TXT)", type=['csv', 'txt'])
    
    if uploaded_chat:
        try:
            file_type = uploaded_chat.name.split('.')[-1].lower()
            
            if file_type == 'csv':
                df = pd.read_csv(uploaded_chat)
            elif file_type == 'txt':
                txt_content = uploaded_chat.read().decode('utf-8-sig', errors='ignore')
                df = parse_whatsapp_txt(txt_content)
            else:
                st.error("❌ Unsupported file")
                df = None
            
            if df is not None and all(col in df.columns for col in ['timestamp', 'sender', 'message']):
                st.success(f"✓ {len(df)} messages loaded")
                
                if st.button("📊 Analyze Chat", use_container_width=True):
                    try:
                        analytics = WhatsAppAnalytics(df)
                        
                        st1, st2, st3, st4 = st.tabs(["📊 Basic Stats", "💬 Who Talks More", "⏱️ Response Patterns", "🕐 Time Activity"])
                        
                        # Basic Stats
                        with st1:
                            basic = analytics.get_basic_stats()
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("💬 Messages", basic['total_messages'])
                            with col2:
                                st.metric("📝 Words", basic['total_words'])
                            with col3:
                                st.metric("📆 Days", basic['chat_duration_days'])
                            with col4:
                                st.metric("📊 Avg", f"{basic['avg_msg_length']:.1f}")
                            
                            user_data = {basic['user1']['name']: basic['user1']['messages'], basic['user2']['name']: basic['user2']['messages']}
                            fig = go.Figure([go.Bar(x=list(user_data.keys()), y=list(user_data.values()), marker_color=['#1f77b4', '#ff7f0e'])])
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Who Talks More
                        with st2:
                            contrib = analytics.get_contribution_stats()
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(f"💬 {contrib['user1']}", contrib['user1_msg_count'])
                                st.metric("📏 Chars", f"{contrib['user1_avg_chars']:.1f}")
                                st.metric("💭 Double", contrib['user1_double_texts'])
                            with col2:
                                st.metric(f"💬 {contrib['user2']}", contrib['user2_msg_count'])
                                st.metric("📏 Chars", f"{contrib['user2_avg_chars']:.1f}")
                                st.metric("💭 Double", contrib['user2_double_texts'])
                            
                            char_data = {contrib['user1']: contrib['user1_avg_chars'], contrib['user2']: contrib['user2_avg_chars']}
                            fig = go.Figure([go.Bar(x=list(char_data.keys()), y=list(char_data.values()), marker_color=['#1f77b4', '#ff7f0e'])])
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Response Patterns
                        with st3:
                            patterns = analytics.get_response_patterns()
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(f"⏱️ {patterns['user1']}", f"{patterns['user1_avg_reply_time']:.1f} min")
                                st.metric("⚡ Fastest", f"{patterns['user1_fastest_reply']:.1f} min")
                            with col2:
                                st.metric(f"⏱️ {patterns['user2']}", f"{patterns['user2_avg_reply_time']:.1f} min")
                                st.metric("⚡ Fastest", f"{patterns['user2_fastest_reply']:.1f} min")
                            
                            col3, col4 = st.columns(2)
                            with col3:
                                st.write(f"🔥 **Max Streak**: {patterns['max_streak_days']} days")
                            with col4:
                                st.write(f"💬 **Longest**: {patterns['longest_conversation']} msgs")
                            
                            reply_times = [patterns['user1_avg_reply_time'], patterns['user2_avg_reply_time']]
                            fig = go.Figure([go.Bar(x=[patterns['user1'], patterns['user2']], y=reply_times, marker_color=['#1f77b4', '#ff7f0e'])])
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Time Activity
                        with st4:
                            time_activity = analytics.get_time_based_activity()
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("🌅 Peak Hour", f"{time_activity['most_active_hour']}:00")
                                st.metric("📊 Count", time_activity['most_active_hour_count'])
                            with col2:
                                st.metric("📅 Peak Day", time_activity['most_active_day'])
                                st.metric("📊 Count", time_activity['most_active_day_count'])
                            
                            if time_activity['hourly_activity']:
                                fig = go.Figure([go.Bar(x=list(time_activity['hourly_activity'].keys()), y=list(time_activity['hourly_activity'].values()), marker_color='#1f77b4')])
                                fig.update_layout(title="📈 By Hour", xaxis_title="Hour", yaxis_title="Messages")
                                st.plotly_chart(fig, use_container_width=True)
                            
                            if time_activity['daily_activity']:
                                fig = go.Figure([go.Bar(x=list(time_activity['daily_activity'].keys()), y=list(time_activity['daily_activity'].values()), marker_color='#ff7f0e')])
                                fig.update_layout(title="📅 By Day", xaxis_title="Day", yaxis_title="Messages")
                                st.plotly_chart(fig, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.error("❌ File must have: timestamp, sender, message")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ==================== TAB 5: ABOUT ====================
with tab5:
    st.header("ℹ️ About")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Features
        - ✨ **Single Text** - Instant sentiment analysis
        - 📊 **Batch** - Hundreds at once
        - 📱 **Chat Analytics** - 4 analytics categories
        - 📈 **Beautiful Charts** - Plotly visualizations
        
        ### 🤖 ML Models
        - **SVM**: 80.74% accuracy
        - **Logistic Regression**: 80.53%
        - **TF-IDF**: 5000 features
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Training
        - **Dataset**: Twitter Airline Sentiment
        - **Size**: 14,640 tweets
        - **Classes**: Positive, Neutral, Negative
        
        ### 📝 File Formats
        - CSV with 'message' column
        - WhatsApp .txt exports
        - Both Supported!
        """)
    
    st.info("🚀 Built with Streamlit + Plotly | 100% Functional")
