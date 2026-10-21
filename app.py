import streamlit as st
import google.generativeai as genai
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Laptop AI Specs & Performance Analyzer",
    page_icon="💻",
    layout="wide"
)

# Sidebar for API Key and Navigation
st.sidebar.title("⚙️ Configuration")
api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")

if api_key:
    genai.configure(api_key=api_key)
    # Using Gemini 1.5 Flash for fast, accurate multimodal/text tasks
    model = genai.GenerativeModel("gemini-1.5-flash")
else:
    st.sidebar.warning("⚠️ Please enter your Gemini API key to use the app.")

st.sidebar.markdown("---")
app_mode = st.sidebar.radio("Navigation", ["🔍 Single Laptop Inspector", "⚔️ Head-to-Head Comparison"])

# Helper function for Gemini calls
def get_gemini_response(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini API: {e}"

# --- TAB 1: SINGLE LAPTOP INSPECTOR ---
if app_mode == "🔍 Single Laptop Inspector":
    st.title("💻 Laptop Hardware & Performance Inspector")
    st.write("Enter the full name or model of a laptop to get a comprehensive breakdown of its hardware, performance ratings, and pricing variations.")

    laptop_name = st.text_input("Enter Laptop Full Name / Model:", placeholder="e.g., ASUS ROG Strix G16 (2024) G614JIR")

    if st.button("Analyze Laptop", type="primary"):
        if not api_key:
            st.error("Please provide your Gemini API key in the sidebar.")
        elif not laptop_name.strip():
            st.warning("Please enter a valid laptop name.")
        else:
            with st.spinner(f"Analyzing specifications and performance for '{laptop_name}'..."):
                prompt = f"""
                You are an expert hardware and gaming laptop reviewer. Analyze the laptop model: "{laptop_name}".
                Provide a detailed breakdown in clean Markdown using the following structure:
                
                ### 📋 Core Hardware Specifications
                - **CPU:** 
                - **GPU:** 
                - **RAM:** (Type, Capacity, Upgradeability)
                - **Storage:** (Type e.g., NVMe SSD, Capacity, Slots)
                - **Display:** (Resolution, Refresh Rate in Hz, Panel Type e.g., IPS, OLED, Brightness)
                - **Battery:** (Capacity in Wh, Expected Battery Life)
                - **Ports:** (List of available ports)
                
                ### 🎮 Gaming Performance
                - **Performance Summary:** 
                - **Estimated FPS in AAA Titles (e.g., Cyberpunk 2077, Modern Warfare III):** 
                - **Thermal & Fan Noise Management:** 
                
                ### 💼 Workstation Performance
                - **Performance Summary (Rendering, Video Editing, Coding, CAD):** 
                - **Pros & Cons for Professionals:** 
                
                ### 💰 Models & Price Variations
                - List the common configurations/sub-models for this laptop series along with their estimated price range (in USD).
                """
                
                result = get_gemini_response(prompt)
                st.markdown(result)

# --- TAB 2: HEAD-TO-HEAD COMPARISON ---
elif app_mode == "⚔️ Head-to-Head Comparison":
    st.title("⚔️ Laptop Head-to-Head Comparison")
    st.write("Compare two different laptop models side-by-side for gaming, workstation tasks, and pricing.")

    col1, col2 = st.columns(2)
    with col1:
        laptop_a = st.text_input("First Laptop Model:", placeholder="e.g., Lenovo Legion Pro 5 Gen 9")
    with col2:
        laptop_b = st.text_input("Second Laptop Model:", placeholder="e.g., Acer Predator Helios 16")

    if st.button("Compare Laptops", type="primary"):
        if not api_key:
            st.error("Please provide your Gemini API key in the sidebar.")
        elif not laptop_a.strip() or not laptop_b.strip():
            st.warning("Please enter both laptop models for comparison.")
        else:
            with st.spinner(f"Comparing '{laptop_a}' vs '{laptop_b}'..."):
                prompt = f"""
                You are an expert hardware reviewer. Compare these two laptops side-by-side:
                1. {laptop_a}
                2. {laptop_b}
                
                Provide a structured comparison in Markdown covering:
                ### 📊 Specification Comparison Matrix
                (Create a clean Markdown table comparing CPU, GPU, RAM, Display, and Battery)
                
                ### 🎮 Gaming Showdown
                - Which laptop performs better in gaming and why?
                
                ### 💼 Workstation Showdown
                - Which laptop is better for creative work, rendering, or coding and why?
                
                ### 💰 Price & Value Comparison
                - Compare their current market pricing tiers and which offers better value for money.
                
                ### 🏆 Final Verdict
                - Which one should the user buy based on different use cases?
                """
                
                result = get_gemini_response(prompt)
                st.markdown(result)
