from openai import OpenAI
import os
from PIL import Image
from runModel import predict_image_class
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def chat():
    client = OpenAI()  # Initialize the OpenAI client once

    def get_ai_response_from_msglist(message_list):
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Changed to a standard model name
            messages=message_list
        )
        return completion.choices[0].message.content
    


    def handle_picture(file_path):
        clean_file_path = file_path.replace('"', '')
        if os.path.isfile(clean_file_path):
            try:
                img = Image.open(clean_file_path)
                img.show()
                print(f"Image successfully loaded. Size: {img.size}")

                probabilities = predict_image_class(img) #put ai results here
                print("Your image results are " + probabilities)
                return probabilities
            except Exception as e:
                print(f"Error opening image: {e}")
        else:
            print("The file does not exist. Please check the path and try again.")
        return None

    def ask(question):
        print()
        print(question)
        return input("You: ")

    knowledge_base = ""
    symptom_list = ""
    reccomendation_list = ""
    diagnosis_from_symptoms = ""
    
    reccomendation_message_list = [
        {"role": "system", "content": "You give reccomendations for the given skin condition. All responses should be maximum 4 sentences/bullet points."}
    ]
    summarizer_message_list = [
        {"role": "system", "content": "You are a doctor. Summarize the reccomendations given by the user into a neat perscriptive summary."}
    ]

    user_response = ask("This app will try to help you with skin problems. Let's get started. Do you want to talk about any skin issues you're facing? (Y/N)")
    if user_response.lower() == 'y':
        user_response = ask("Have you been diagnosed for this skin issue?")

        if user_response.lower() == 'n':
            user_response = ask("describe your symptoms")
            symptom_list += "Symptoms: " + user_response
            diagnose_message_list = [
                {"role": "system", "content": "You are a doctor. Give potential diagnosies on what the skin condition could be based on the given symptoms. Do not give reccomendations for what to do about it."},
                {"role": "user", "content": symptom_list}
            ]
            diagnosis_from_symptoms = get_ai_response_from_msglist(diagnose_message_list)
            print(diagnosis_from_symptoms)

            file_path = ask("Please upload the file path of an image of your skin condition")
            image_result = handle_picture(file_path)
            if image_result:
                knowledge_base += image_result

    user_response = ask("Would you like to get recommendations for a skin disease? Type 'N' or type the skin disease you would like to ask about.")
    if user_response.lower() != 'n':
        reccomendation_message_list.append({"role": "user", "content": "I have " + user_response})
        while(True):
            ai_response = get_ai_response_from_msglist(reccomendation_message_list)
            reccomendation_message_list.append({"role": "assistant", "content": ai_response})
            print()
            print(ai_response)
            reccomendation_list += ai_response + "\n"
            user_response = ask("Continue chatting or press Q to move on")
            if user_response.lower() == 'q':
                break
            reccomendation_message_list.append({"role": "user", "content": user_response})

            
            


            
    summarizer_message_list.append({"role": "user", "content": reccomendation_list})
    print("Thanks for using derm AI")
    print("Here is your diagnosis:")
    print(diagnosis_from_symptoms)
    print()
    print("here are your recommendations:")
    print(get_ai_response_from_msglist(summarizer_message_list))

chat()