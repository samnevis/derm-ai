#main.py
from openai import OpenAI
import os
from PIL import Image
import sys
import os

# Add ../CNN to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../CNN")))

# Now you can import predict_image_class
from runModel import predict_image_class

# Disable ONEDNN optimization for TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def chat():
    # Initialize the OpenAI client
    client = OpenAI()

    # Function to get AI response from a list of messages
    def get_ai_response_from_msglist(message_list):
        # Send the chat messages to the OpenAI API and get the completion response
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use the gpt-3.5-turbo model
            messages=message_list
        )
        # Return the content of the first choice's message
        return completion.choices[0].message.content
    
    # Function to handle image input, analyze it, and display predictions
    def handle_picture(file_path):
        # Clean the file path by removing any extra quotes
        clean_file_path = file_path.replace('"', '')
        
        # Check if the file exists
        if os.path.isfile(clean_file_path):
            try:
                # Open the image using PIL
                img = Image.open(clean_file_path)
                img.show()  # Display the image to the user
                print(f"Image successfully loaded. Size: {img.size}")

                # Run the image through the prediction model
                probabilities = predict_image_class(img)
                print("\n\nYour image results are " + probabilities)
                
                return probabilities
            except Exception as e:
                # Handle any errors in opening or processing the image
                print(f"Error opening image: {e}")
        else:
            # Notify the user if the file does not exist
            print("The file does not exist. Please check the path and try again.")
        
        return None

    # Function to prompt the user for input and return their response
    def ask(question):
        print("===============================================")
        print(question)
        print()
        return input("You: ")

    # Initialize variables for storing conversation data
    knowledge_base = ""
    symptom_list = ""
    reccomendation_list = ""
    diagnosis_from_symptoms = ""
    
    # Initialize message lists for recommendations and summary
    reccomendation_message_list = [
        {"role": "system", "content": "You give recommendations for the given skin condition. All responses should be maximum 4 sentences or bullet points."}
    ]
    
    summarizer_message_list = [
        {"role": "system", "content": "You are a doctor. Summarize the recommendations given by the user into a neat prescriptive summary."}
    ]

    # Start the conversation

    user_response = ask("\n\nThis app will try to help you with skin problems. Let's get started. Do you want to talk about any skin issues you're facing? (Y/N)")
    
    if user_response.lower() == 'y':
        user_response = ask("Have you been diagnosed for this skin issue?")

        # If the user has not been diagnosed, ask for symptoms
        if user_response.lower() == 'n':
            user_response = ask("Describe your symptoms")
            symptom_list += "Symptoms: " + user_response
            
            # Generate a potential diagnosis based on the user's symptoms
            diagnose_message_list = [
                {"role": "system", "content": "You are a doctor. Give potential diagnoses on what the skin condition could be based on the given symptoms. Do not give recommendations for what to do about it."},
                {"role": "user", "content": symptom_list}
            ]
            
            # Get the diagnosis from the AI
            diagnosis_from_symptoms = get_ai_response_from_msglist(diagnose_message_list)
            print(diagnosis_from_symptoms)

            # Ask the user to provide an image for further analysis
            file_path = ask("Please upload the file path of an image of your skin condition")
            image_result = handle_picture(file_path)
            
            # Append image analysis results to the knowledge base if any
            if image_result:
                knowledge_base += image_result

    # Ask if the user wants recommendations for a specific skin disease
    user_response = ask("Would you like to get recommendations for a skin disease? Type 'N' or type the skin disease you would like to ask about.")
    
    if user_response.lower() != 'n':
        reccomendation_message_list.append({"role": "user", "content": "I have " + user_response})
        
        # Continue the conversation until the user decides to stop
        while True:
            # Get AI recommendations based on the skin disease
            ai_response = get_ai_response_from_msglist(reccomendation_message_list)
            reccomendation_message_list.append({"role": "assistant", "content": ai_response})
            print()
            print(ai_response)
            
            # Add the recommendations to the summary list
            reccomendation_list += ai_response + "\n"
            
            # Ask the user if they want to continue or quit
            user_response = ask("Continue chatting or press Q to move on")
            if user_response.lower() == 'q':
                break
            # Append user's follow-up input to the message list
            reccomendation_message_list.append({"role": "user", "content": user_response})
    else:
        print("\n\n\nThanks for using derm AI")
        print("Here is your diagnosis:")
        print(diagnosis_from_symptoms)
        print()
        return None
         

    # Generate a summary of the recommendations
    summarizer_message_list.append({"role": "user", "content": reccomendation_list})
    
    # Print the final output to the user
    print("\n\n\nThanks for using derm AI")
    print()
    print("Here is your diagnosis:")
    print()
    print(diagnosis_from_symptoms)
    print()
    print("Here are your recommendations:")
    print()
    print(get_ai_response_from_msglist(summarizer_message_list))
    print()

# Run the chat function when the script is executed
chat()
