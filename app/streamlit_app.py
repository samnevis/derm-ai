import streamlit as st
from openai import OpenAI
import os
from PIL import Image
import sys
import io

# Add CNN to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../CNN")))
from runModel import predict_image_class

# Disable ONEDNN optimization for TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize OpenAI client
client = OpenAI()

# Set up the Streamlit page
st.set_page_config(page_title="DermAI - Skin Condition Assistant", layout="wide")
st.title("DermAI - Your Skin Health Assistant")
st.markdown("This application helps identify potential skin conditions and provides recommendations.")
st.warning("NOTE: This app is for informational purposes only and does not replace professional medical advice.")

# Function to get AI response
def get_ai_response_from_msglist(message_list):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=message_list
    )
    return completion.choices[0].message.content

# Initialize session state variables if they don't exist
if 'symptom_list' not in st.session_state:
    st.session_state.symptom_list = ""
if 'diagnosis' not in st.session_state:
    st.session_state.diagnosis = ""
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = ""
if 'image_result' not in st.session_state:
    st.session_state.image_result = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Diagnosis", "Recommendations", "Summary"])

# Add this function before your main code
def format_prediction_result(prediction_string):
    """Convert the raw prediction string to a user-friendly format"""
    try:
        # Debug the raw input
        print(f"Raw prediction: {prediction_string}")
        
        # Parse the prediction string
        predictions = {}
        parts = prediction_string.split('%')
        
        for part in parts:
            if ':' in part:
                # Handle the format "Condition: XX.XX%"
                condition_percent = part.strip()
                if condition_percent:
                    condition, percent_str = condition_percent.rsplit(':', 1)
                    condition = condition.strip()
                    try:
                        percent = float(percent_str.strip())
                        predictions[condition] = percent
                    except ValueError:
                        # Skip if percent can't be converted to float
                        pass
        
        # If no predictions were parsed, try another approach
        if not predictions and ":" in prediction_string:
            items = prediction_string.split()
            for i in range(len(items)-1):
                if items[i].endswith(':') and items[i+1].endswith('%'):
                    condition = items[i][:-1].strip()
                    percent = float(items[i+1][:-1].strip())
                    predictions[condition] = percent
        
        # Find the condition with highest percentage
        if predictions:
            top_condition = max(predictions.items(), key=lambda x: x[1])
            condition_name = top_condition[0]
            confidence = top_condition[1]
            
            # Format the message
            result = f"Our AI model suggests this might be **{condition_name}** (confidence: {confidence:.1f}%)."
            
            # Add disclaimer
            result += "\n\n*Note: This is only an AI estimate and should not replace professional medical diagnosis.*"
            
            return result
        else:
            # If we still couldn't parse, let's just display the top condition directly
            # This is a fallback approach
            lines = prediction_string.split('\n')
            for line in lines:
                if ':' in line and '%' in line:
                    parts = line.split(':')
                    condition = parts[0].strip()
                    percent = parts[1].strip()
                    if percent.endswith('%'):
                        return f"Our AI model suggests this might be **{condition}** (confidence: {percent})."
            
            # If all else fails, just show the raw output with a friendly message
            return f"Our AI model analysis: {prediction_string}\n\n*Note: This is only an AI estimate and should not replace professional medical diagnosis.*"
    except Exception as e:
        # Print the exception for debugging
        import traceback
        print(f"Error in format_prediction_result: {str(e)}")
        print(traceback.format_exc())
        
        # Return a user-friendly message with the raw prediction
        return f"Our AI model analysis: {prediction_string}\n\n*Note: This is only an AI estimate and should not replace professional medical diagnosis.*"

# Add this function after your format_prediction_result function
def get_condition_explanation(condition_name):
    """Get an explanation of the skin condition from the AI"""
    explanation_message_list = [
        {"role": "system", "content": "You are a dermatologist. Provide a brief, clear explanation of the skin condition in 3-4 sentences. Include common symptoms, causes, and general outlook."},
        {"role": "user", "content": f"Explain {condition_name} in simple terms."}
    ]
    return get_ai_response_from_msglist(explanation_message_list)

# Add this function to relate the image diagnosis to the symptom diagnosis
def relate_diagnoses(image_condition, symptom_diagnosis):
    """Relate the image-based diagnosis to the symptom-based diagnosis"""
    if not symptom_diagnosis:
        return ""
        
    relation_message_list = [
        {"role": "system", "content": "You are a dermatologist. Compare the AI image diagnosis with the symptom-based diagnosis and explain if they align or differ. Be brief (2-3 sentences)."},
        {"role": "user", "content": f"The AI image analysis suggests {image_condition}. The symptom-based diagnosis was: {symptom_diagnosis}. Do these align? Explain briefly."}
    ]
    return get_ai_response_from_msglist(relation_message_list)

with tab1:
    st.header("Describe Your Skin Condition")
    
    # Ask if diagnosed
    diagnosed = st.radio("Have you been diagnosed for this skin issue?", ["Yes", "No"])
    
    if diagnosed == "No":
        # Get symptoms
        symptoms = st.text_area("Describe your symptoms in detail:")
        if st.button("Submit Symptoms"):
            st.session_state.symptom_list = "Symptoms: " + symptoms
            
            # Generate diagnosis
            diagnose_message_list = [
                {"role": "system", "content": "You are a doctor. Give potential diagnoses on what the skin condition could be based on the given symptoms. Do not give recommendations for what to do about it."},
                {"role": "user", "content": st.session_state.symptom_list}
            ]
            
            with st.spinner("Analyzing symptoms..."):
                st.session_state.diagnosis = get_ai_response_from_msglist(diagnose_message_list)
            
            st.subheader("Potential Diagnosis")
            st.write(st.session_state.diagnosis)
        
        # Image upload
        st.subheader("Upload an Image")
        uploaded_file = st.file_uploader("Upload an image of your skin condition", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=300)
            
            if st.button("Analyze Image"):
                with st.spinner("Analyzing image..."):
                    try:
                        raw_prediction = predict_image_class(image)
                        st.session_state.image_result = raw_prediction
                        
                        # Manual parsing for the specific format
                        highest_condition = ""
                        highest_percent = 0
                        
                        # Try to parse the string
                        try:
                            # Split by spaces and process each part
                            parts = raw_prediction.split()
                            current_condition = None
                            
                            for part in parts:
                                if part.endswith(':'):
                                    # This is a condition name
                                    current_condition = part[:-1]
                                elif part.endswith('%') and current_condition:
                                    # This is a percentage
                                    try:
                                        percent = float(part[:-1])
                                        if percent > highest_percent:
                                            highest_percent = percent
                                            highest_condition = current_condition
                                    except ValueError:
                                        pass
                        
                            if highest_condition:
                                # Store the diagnosis in session state
                                st.session_state.image_diagnosis = {
                                    "condition": highest_condition,
                                    "confidence": highest_percent
                                }
                                
                                # Display the main diagnosis
                                st.markdown(f"### Image Analysis Result")
                                st.markdown(f"Our AI model suggests this might be **{highest_condition}** (confidence: {highest_percent:.1f}%).")
                                
                                # Get and display explanation of the condition
                                with st.spinner("Getting condition information..."):
                                    explanation = get_condition_explanation(highest_condition)
                                    st.markdown("### About this Condition")
                                    st.markdown(explanation)
                                
                                # Relate to symptom-based diagnosis if available
                                if st.session_state.diagnosis:
                                    with st.spinner("Comparing with symptom analysis..."):
                                        relation = relate_diagnoses(highest_condition, st.session_state.diagnosis)
                                        st.markdown("### Comparison with Symptom Analysis")
                                        st.markdown(relation)
                                
                                # Add disclaimer
                                st.markdown("*Note: This is only an AI estimate and should not replace professional medical diagnosis.*")
                            else:
                                st.warning("Could not determine the most likely condition. Please see the detailed analysis below.")
                        except Exception as e:
                            st.warning(f"Error parsing prediction: {str(e)}")
                            st.markdown("Our AI has analyzed your image. Please see the detailed results below.")
                        
                        # Always show the raw analysis in an expander
                        with st.expander("See detailed analysis"):
                            st.text(raw_prediction)
                            
                    except Exception as e:
                        st.error(f"Error analyzing image: {str(e)}")
                        st.info("Please try another image or check if the image is clear enough.")

with tab2:
    st.header("Get Recommendations")
    
    condition = st.text_input("What skin condition would you like recommendations for?")
    
    if st.button("Get Recommendations") and condition:
        reccomendation_message_list = [
            {"role": "system", "content": "You give recommendations for the given skin condition. All responses should be maximum 4 sentences or bullet points."},
            {"role": "user", "content": "I have " + condition}
        ]
        
        with st.spinner("Generating recommendations..."):
            recommendations = get_ai_response_from_msglist(reccomendation_message_list)
            st.session_state.recommendations = recommendations
            st.session_state.chat_history.append(("AI", recommendations))
        
        st.write(recommendations)
    
    # Chat interface for follow-up questions
    st.subheader("Ask Follow-up Questions")
    user_input = st.text_input("Your question:")
    
    if st.button("Send") and user_input:
        st.session_state.chat_history.append(("User", user_input))
        
        reccomendation_message_list = [
            {"role": "system", "content": "You give recommendations for skin conditions. All responses should be maximum 4 sentences or bullet points."}
        ]
        
        # Add chat history to message list
        for role, content in st.session_state.chat_history:
            if role == "User":
                reccomendation_message_list.append({"role": "user", "content": content})
            else:
                reccomendation_message_list.append({"role": "assistant", "content": content})
        
        with st.spinner("Thinking..."):
            ai_response = get_ai_response_from_msglist(reccomendation_message_list)
            st.session_state.chat_history.append(("AI", ai_response))
            st.session_state.recommendations += "\n" + ai_response
        
    # Display chat history
    for role, content in st.session_state.chat_history:
        if role == "User":
            st.write(f"**You:** {content}")
        else:
            st.write(f"**AI:** {content}")

with tab3:
    st.header("Summary")
    
    # Display symptom-based diagnosis if available
    if st.session_state.diagnosis:
        st.subheader("Symptom-Based Diagnosis")
        st.write(st.session_state.diagnosis)
    
    # Display image-based diagnosis if available
    if hasattr(st.session_state, 'image_diagnosis'):
        st.subheader("Image-Based Diagnosis")
        condition = st.session_state.image_diagnosis["condition"]
        confidence = st.session_state.image_diagnosis["confidence"]
        st.markdown(f"Our AI model identified this as likely **{condition}** (confidence: {confidence:.1f}%).")
        
        # Get a brief explanation if not already displayed
        if 'image_explanation' not in st.session_state:
            with st.spinner("Loading condition information..."):
                st.session_state.image_explanation = get_condition_explanation(condition)
        
        st.markdown(st.session_state.image_explanation)
    
    # Display recommendations if available
    if st.session_state.recommendations:
        st.subheader("Recommendations")
        
        # Include image diagnosis in the summary if available
        summary_content = st.session_state.recommendations
        if hasattr(st.session_state, 'image_diagnosis'):
            condition = st.session_state.image_diagnosis["condition"]
            summary_content = f"Based on image analysis showing possible {condition}. " + summary_content
        
        summarizer_message_list = [
            {"role": "system", "content": "You are a doctor. Summarize the recommendations given by the user into a neat prescriptive summary."},
            {"role": "user", "content": summary_content}
        ]
        
        with st.spinner("Generating summary..."):
            summary = get_ai_response_from_msglist(summarizer_message_list)
        
        st.write(summary)
        
        # Add download button for summary with all information
        download_content = ""
        if st.session_state.diagnosis:
            download_content += f"SYMPTOM-BASED DIAGNOSIS:\n{st.session_state.diagnosis}\n\n"
        
        if hasattr(st.session_state, 'image_diagnosis'):
            condition = st.session_state.image_diagnosis["condition"]
            confidence = st.session_state.image_diagnosis["confidence"]
            download_content += f"IMAGE-BASED DIAGNOSIS:\n{condition} (confidence: {confidence:.1f}%)\n"
            if 'image_explanation' in st.session_state:
                download_content += f"{st.session_state.image_explanation}\n\n"
        
        download_content += f"RECOMMENDATIONS:\n{summary}"
        
        st.download_button(
            label="Download Complete Summary",
            data=download_content,
            file_name="dermai_summary.txt",
            mime="text/plain"
        ) 