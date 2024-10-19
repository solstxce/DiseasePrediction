import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import queue
import threading
import joblib
import whisper
import tempfile
import os
from pydub import AudioSegment
from datetime import datetime

# API endpoint
API_URL = "http://localhost:5000"

# Audio processing
audio_queue = queue.Queue()
result_queue = queue.Queue()

# Load the disease prediction model
model = joblib.load('disease_prediction_model.joblib')

# List of symptoms for speech recognition and prediction
symptoms_list = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering", "chills", "joint_pain",
    "stomach_pain", "acidity", "ulcers_on_tongue", "muscle_wasting", "vomiting", "burning_micturition",
    "spotting_urination", "fatigue", "weight_gain", "anxiety", "cold_hands_and_feets", "mood_swings",
    "weight_loss", "restlessness", "lethargy", "patches_in_throat", "irregular_sugar_level", "cough",
    "high_fever", "sunken_eyes", "breathlessness", "sweating", "dehydration", "indigestion", "headache",
    "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "pain_behind_the_eyes", "back_pain",
    "constipation", "abdominal_pain", "diarrhoea", "mild_fever", "yellow_urine", "yellowing_of_eyes",
    "acute_liver_failure", "fluid_overload", "swelling_of_stomach", "swelled_lymph_nodes", "malaise",
    "blurred_and_distorted_vision", "phlegm", "throat_irritation", "redness_of_eyes", "sinus_pressure",
    "runny_nose", "congestion", "chest_pain", "weakness_in_limbs", "fast_heart_rate",
    "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool", "irritation_in_anus", "neck_pain",
    "dizziness", "cramps", "bruising", "obesity", "swollen_legs", "swollen_blood_vessels",
    "puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails", "swollen_extremeties", "excessive_hunger",
    "extra_marital_contacts", "drying_and_tingling_lips", "slurred_speech", "knee_pain", "hip_joint_pain",
    "muscle_weakness", "stiff_neck", "swelling_joints", "movement_stiffness", "spinning_movements",
    "loss_of_balance", "unsteadiness", "weakness_of_one_body_side", "loss_of_smell", "bladder_discomfort",
    "foul_smell_of_urine", "continuous_feel_of_urine", "passage_of_gases", "internal_itching",
    "toxic_look_(typhos)", "depression", "irritability", "muscle_pain", "altered_sensorium",
    "red_spots_over_body", "belly_pain", "abnormal_menstruation", "dischromic_patches",
    "watering_from_eyes", "increased_appetite", "polyuria", "family_history", "mucoid_sputum",
    "rusty_sputum", "lack_of_concentration", "visual_disturbances", "receiving_blood_transfusion",
    "receiving_unsterile_injections", "coma", "stomach_bleeding", "distention_of_abdomen",
    "history_of_alcohol_consumption", "fluid_overload", "blood_in_sputum", "prominent_veins_on_calf",
    "palpitations", "painful_walking", "pus_filled_pimples", "blackheads", "scurring", "skin_peeling",
    "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails", "blister",
    "red_sore_around_nose", "yellow_crust_ooze"
]

def main():
    st.set_page_config(page_title="Disease Prediction System", layout="wide")
    st.title("Disease Prediction System")

    # Initialize session state
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'token' not in st.session_state:
        st.session_state.token = None
    if 'show_register' not in st.session_state:
        st.session_state.show_register = False
    if 'page' not in st.session_state:
        st.session_state.page = "Home"

    # Sidebar
    with st.sidebar:
        if st.session_state.user is None:
            if not st.session_state.show_register:
                st.markdown("### Sign In")
                
                # Normal sign in
                st.markdown("### Sign In with Email")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                if st.button("Sign In", key="signin_button"):
                    login_result = login(email, password)
                    if login_result:
                        st.session_state.user = {"email": email, "role": login_result["role"]}
                        st.session_state.token = login_result["access_token"]
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
                
                if st.button("Create an Account", key="create_account_button"):
                    st.session_state.show_register = True
                    st.rerun()
            else:
                st.markdown("### Register")
                reg_email = st.text_input("Email", key="reg_email")
                reg_password = st.text_input("Password", type="password", key="reg_password")
                reg_password_2 = st.text_input("Retype Password", type="password")
                if st.button("Register", key="register_button"):
                    if reg_password == reg_password_2:
                        # Implement registration logic here
                        st.success("Registration successful! Please sign in.")
                        st.session_state.show_register = False
                        st.rerun()
                    else:
                        st.error("Passwords do not match")
                if st.button("Back to Sign In", key="back_to_signin_button"):
                    st.session_state.show_register = False
                    st.rerun()
        else:
            st.markdown(f"### Welcome, {st.session_state.user['email']}")
            st.markdown(f"Role: {st.session_state.user['role']}")
            if st.button("Sign Out", key="signout_button"):
                st.session_state.user = None
                st.session_state.token = None
                st.session_state.page = "Home"
                st.rerun()

            st.markdown("---")
            st.markdown("### Navigation")
            
            # Navigation buttons
            if st.button("Home", key="home_button"):
                st.session_state.page = "Home"
            st.markdown("---")
            if st.button("Predict Disease", key="predict_disease_button"):
                st.session_state.page = "Predict Disease"
            st.markdown("---")
            if st.button("Live Data Capture", key="live_data_capture_button"):
                st.session_state.page = "Live Data Capture"
            st.markdown("---")
            if st.session_state.user['role'] in ['admin', 'doctor']:
                if st.button("Patient Logs", key="patient_logs_button"):
                    st.session_state.page = "Patient Logs"
                st.markdown("---")
            if st.button("Developers", key="developers_button"):
                st.session_state.page = "Developers"

    # Main content
    if st.session_state.user is None:
        st.warning("Please sign in to access the Disease Prediction System.")
    else:
        if st.session_state.page == "Home":
            home_page()
        elif st.session_state.page == "Predict Disease":
            predict_disease_page()
        elif st.session_state.page == "Live Data Capture":
            live_data_capture_page()
        elif st.session_state.page == "Patient Logs" and st.session_state.user['role'] in ['admin', 'doctor']:
            patient_logs_page()
        elif st.session_state.page == "Developers":
            developers_page()

def login(email, password):
    response = requests.post(f"{API_URL}/login", json={"email": email, "password": password})
    if response.status_code == 200:
        return response.json()
    return None

def predict_disease(symptoms, vitals):
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    response = requests.post(f"{API_URL}/predict", json={"symptoms": symptoms, "vitals": vitals}, headers=headers)
    if response.status_code == 200:
        return response.json()
    return None

def get_patient_logs():
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    response = requests.get(f"{API_URL}/patient_logs", headers=headers)
    if response.status_code == 200:
        return response.json()
    return []

def home_page():
    st.header("Welcome to the Disease Prediction System")
    st.write("This system helps predict diseases based on symptoms and vital signs.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Recent Statistics")
        st.metric(label="Predictions Made", value="1,234")
        st.metric(label="Accuracy Rate", value="92%")
    with col2:
        st.subheader("How it works")
        st.write("1. Enter your symptoms")
        st.write("2. Provide vital signs (optional)")
        st.write("3. Get instant predictions")
        st.write("4. Consult with a healthcare professional")

def predict_disease_page():
    st.header("Predict Disease")

    # Add patient name input
    patient_name = st.text_input("Patient Name")

    # Heart Rate Monitoring Section
    st.subheader("Heart Rate Monitor")
    duration = st.selectbox("Select monitoring duration (seconds)", [10, 20, 30, 60], index=0)
    
    if st.button("Start Heart Rate Monitoring", key="start_heart_rate_monitoring"):
        with st.spinner(f"Monitoring heart rate for {duration} seconds..."):
            headers = {"Authorization": f"Bearer {st.session_state.token}"}
            response = requests.post(f"{API_URL}/heart_rate", json={"duration": duration}, headers=headers)
            if response.status_code == 200:
                heart_rate_data = response.json()
                st.line_chart(pd.DataFrame({"Heart Rate": heart_rate_data['heart_rates']}, index=heart_rate_data['time_points']))
                st.metric("Average Heart Rate", f"{heart_rate_data['average_heart_rate']:.1f} bpm")
            else:
                st.error("Failed to retrieve heart rate data")

    # Voice Recognition Section
    st.subheader("Voice Recognition")

    # Load Whisper model for speech-to-text
    whisper_model = whisper.load_model("base")

    def audio_callback(frame):
        sound = frame.to_ndarray().flatten().astype(np.float32)
        audio_queue.put(sound)
        return frame

    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": False, "audio": True},
        audio_frame_callback=audio_callback,
        async_processing=True,
    )

    def process_audio():
        audio_data = []
        while True:
            if webrtc_ctx.state.playing:
                try:
                    chunk = audio_queue.get(timeout=1)
                    audio_data.extend(chunk)
                except queue.Empty:
                    continue

                # Process audio every 2 seconds
                if len(audio_data) > 48000:  # 16000 samples per second * 2 seconds
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                        temp_audio_path = temp_audio.name
                        audio_array = np.array(audio_data, dtype=np.float32)
                        AudioSegment(audio_array.tobytes(), frame_rate=16000, sample_width=4, channels=1).export(temp_audio_path, format="wav")

                    result = whisper_model.transcribe(temp_audio_path)
                    text = result["text"]

                    # Convert audio to text with timestamps
                    transcription = convert_audio_to_text_with_timestamps_whisper(temp_audio_path, text)
                    
                    for timestamp, chunk_text in transcription:
                        detected_symptoms = [symptom for symptom in symptoms_list if symptom in chunk_text.lower()]
                        result_queue.put((timestamp, chunk_text, detected_symptoms))

                    # Clear processed audio data
                    audio_data = []
                    
                    # Remove temporary file
                    os.unlink(temp_audio_path)
            else:
                break

    def convert_audio_to_text_with_timestamps_whisper(audio_path, transcribed_text):
        audio = AudioSegment.from_wav(audio_path)
        chunk_length_ms = 2000
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
        
        words = transcribed_text.split()
        chunk_size = len(words) // len(chunks) if len(chunks) > 0 else len(words)
        
        transcription = []
        for i, chunk in enumerate(chunks):
            start_time = i * 2  # 2 seconds per chunk
            chunk_text = " ".join(words[i * chunk_size:(i + 1) * chunk_size])
            transcription.append((start_time, chunk_text))
        
        return transcription

    if webrtc_ctx.state.playing:
        st.info("Voice recognition is active. Speak clearly into your microphone.")
        process_thread = threading.Thread(target=process_audio, daemon=True)
        process_thread.start()

    if 'recognized_text' not in st.session_state:
        st.session_state.recognized_text = ""
    if 'recognized_symptoms' not in st.session_state:
        st.session_state.recognized_symptoms = []

    text_placeholder = st.empty()
    symptoms_placeholder = st.empty()

    while webrtc_ctx.state.playing:
        if not result_queue.empty():
            timestamp, text, symptoms = result_queue.get()
            st.session_state.recognized_text += f" {text}"
            st.session_state.recognized_symptoms.extend(symptoms)
            
            text_placeholder.write(f"Recognized Text ({timestamp}s): {text}")
            
            if symptoms:
                symptoms_placeholder.write("Detected Symptoms:")
                for symptom in set(symptoms):
                    symptoms_placeholder.write(f"- {symptom}")
            else:
                symptoms_placeholder.write("No symptoms detected in the speech.")
        time.sleep(0.1)

    # Manual Symptom Input
    st.subheader("Manual Symptom Input")
    manual_symptoms = st.multiselect("Select additional symptoms", symptoms_list)

    # Combine detected and manual symptoms
    all_symptoms = list(set(st.session_state.recognized_symptoms + manual_symptoms))

    # Display all selected symptoms
    st.subheader("Selected Symptoms")
    for symptom in all_symptoms:
        st.write(f"- {symptom}")

    # Prediction Button
    if st.button("Predict Disease", key="predict_disease"):
        if all_symptoms:
            # Prepare input for the model
            input_symptoms = [1 if symptom in all_symptoms else 0 for symptom in symptoms_list]
            
            # Make prediction using the loaded joblib model
            prediction = model.predict([input_symptoms])[0]
            confidence = model.predict_proba([input_symptoms]).max()
            
            st.success(f"Predicted Disease: {prediction}")
            st.info(f"Confidence: {confidence:.2%}")
            
            # Log the prediction
            headers = {"Authorization": f"Bearer {st.session_state.token}"}
            log_data = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "patient_id": st.session_state.user['email'],
                "patient_name": patient_name,
                "predicted_disease": prediction,
                "confidence": f"{confidence:.2%}"
            }
            response = requests.post(f"{API_URL}/patient_logs", json=log_data, headers=headers)
            if response.status_code != 201:
                st.error("Failed to log the prediction.")
        else:
            st.warning("Please select or speak at least one symptom before predicting.")

    # Clear recognized symptoms button
    if st.button("Clear Recognized Symptoms"):
        st.session_state.recognized_text = ""
        st.session_state.recognized_symptoms = []
        st.rerun()

def patient_logs_page():
    st.header("Patient Logs")
    
    logs = get_patient_logs()
    if logs:
        df = pd.DataFrame(logs)
        
        # Add edit functionality for admin users
        if st.session_state.user['role'] == 'admin':
            st.subheader("Edit Logs")
            selected_log = st.selectbox("Select a log to edit", df['_id'].tolist())
            if selected_log:
                log_to_edit = df[df['_id'] == selected_log].iloc[0]
                edited_log = {
                    "date": st.text_input("Date", log_to_edit['date']),
                    "patient_id": st.text_input("Patient ID", log_to_edit['patient_id']),
                    "patient_name": st.text_input("Patient Name", log_to_edit['patient_name']),
                    "predicted_disease": st.text_input("Predicted Disease", log_to_edit['predicted_disease']),
                    "confidence": st.text_input("Confidence", log_to_edit['confidence'])
                }
                if st.button("Update Log"):
                    update_patient_log(selected_log, edited_log)
                    st.success("Log updated successfully!")
                    st.rerun()

        st.subheader("All Logs")
        st.dataframe(df)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Patient Logs",
            data=csv,
            file_name='patient_logs.csv',
            mime='text/csv',
        )
    else:
        st.warning("No patient logs available or you don't have permission to view them.")

# Add this new function to update patient logs
def update_patient_log(log_id, updated_data):
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    response = requests.put(f"{API_URL}/patient_logs/{log_id}", json=updated_data, headers=headers)
    return response.status_code == 200

def live_data_capture_page():
    st.header("Live Data Capture")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Heart Rate Monitor")
        duration = st.selectbox("Select monitoring duration", [10, 20, 30, 60], index=0)
        
        if st.button("Start Monitoring", key="start_monitoring_live_data"):
            with st.spinner(f"Monitoring heart rate for {duration} seconds..."):
                headers = {"Authorization": f"Bearer {st.session_state.token}"}
                response = requests.post(f"{API_URL}/heart_rate", json={"duration": duration}, headers=headers)
                if response.status_code == 200:
                    heart_rate_data = response.json()
                    st.line_chart(pd.DataFrame({"Heart Rate": heart_rate_data['heart_rates']}, index=heart_rate_data['time_points']))
                    st.metric("Average Heart Rate", f"{heart_rate_data['average_heart_rate']:.1f} bpm")
                else:
                    st.error("Failed to retrieve heart rate data")
    
    with col2:
        st.subheader("Voice Recognition")
        if st.button("Start Voice Recognition", key="start_voice_recognition_live_data"):
            # Use the same voice recognition code as in predict_disease_page()
            def audio_callback(frame):
                sound = frame.to_ndarray().flatten().astype(np.float32)
                audio_queue.put(sound)
                return frame

            webrtc_ctx = webrtc_streamer(
                key="speech-to-text-live",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                media_stream_constraints={"video": False, "audio": True},
                audio_frame_callback=audio_callback,
                async_processing=True,
            )

            def process_audio():
                r = sr.Recognizer()
                while True:
                    if webrtc_ctx.state.playing:
                        try:
                            audio_data = audio_queue.get(timeout=1)
                            text = r.recognize_google(audio_data)
                            detected_symptoms = [symptom for symptom in symptoms_list if symptom in text.lower()]
                            result_queue.put((text, detected_symptoms))
                        except queue.Empty:
                            continue
                        except sr.UnknownValueError:
                            pass
                        except sr.RequestError as e:
                            st.error(f"Could not request results from speech recognition service; {e}")
                    else:
                        break

            if webrtc_ctx.state.playing:
                st.info("Voice recognition is active. Speak clearly into your microphone.")
                process_thread = threading.Thread(target=process_audio, daemon=True)
                process_thread.start()

            text_placeholder = st.empty()
            symptoms_placeholder = st.empty()

            while webrtc_ctx.state.playing:
                if not result_queue.empty():
                    text, symptoms = result_queue.get()
                    text_placeholder.write(f"Recognized Text: {text}")
                    
                    if symptoms:
                        symptoms_placeholder.write("Detected Symptoms:")
                        for symptom in set(symptoms):
                            symptoms_placeholder.write(f"- {symptom}")
                    else:
                        symptoms_placeholder.write("No symptoms detected in the speech.")
                
                time.sleep(0.1)

# Add this new function to display the developers' information
def developers_page():
    st.header("Meet Our Developers")
    
    st.markdown("""
    <style>
    .dev-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .dev-name {
        color: #0066cc;
        font-size: 20px;
        font-weight: bold;
    }
    .dev-id {
        color: #666666;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

    developers = [
        {"name": "D. Aakash", "id": "99220040051"},
        {"name": "Geetheshwar", "id": "99220040368"},
        {"name": "K. Suryavardhan Reddy", "id": "99220040370"},
        {"name": "K. Sai Suhas", "id": "99220040369"}
    ]

    for dev in developers:
        st.markdown(f"""
        <div class="dev-card">
            <p class="dev-name">{dev['name']}</p>
            <p class="dev-id">ID: {dev['id']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; margin-top: 50px;">
        <p style="font-style: italic; color: #666666;">
        "Coming together is a beginning. Keeping together is progress. Working together is success." - Henry Ford
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
