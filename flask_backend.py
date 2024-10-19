from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from functools import wraps
import random
import time
from pymongo import MongoClient
from bson import ObjectId
import numpy as np

app = Flask(__name__)
api = Api(app)
app.config['JWT_SECRET_KEY'] = 'your-secret-key'  # Change this!
jwt = JWTManager(app)

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['disease_prediction']
users_collection = db['users']
patient_logs_collection = db['patient_logs']

# Update the symptoms list to be more comprehensive
symptoms_list = [
    "fever", "cough", "fatigue", "shortness of breath", "headache", "loss of taste", "loss of smell",
    "sore throat", "runny nose", "nausea", "vomiting", "diarrhea", "body aches", "chills",
    "dizziness", "chest pain", "congestion", "muscle pain", "rash", "abdominal pain"
]

def admin_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        current_user = get_jwt_identity()
        if users_collection.find_one({'email': current_user})['role'] not in ['admin', 'doctor']:
            return jsonify({"msg": "Admin or doctor access required"}), 403
        return fn(*args, **kwargs)
    return wrapper

class UserLogin(Resource):
    def post(self):
        email = request.json.get('email', None)
        password = request.json.get('password', None)
        user = users_collection.find_one({'email': email})
        if not user or user['password'] != password:
            return {"message": "Invalid credentials"}, 401
        access_token = create_access_token(identity=email)
        return {"access_token": access_token, "role": user['role']}

class PatientLogs(Resource):
    @jwt_required()
    @admin_required
    def get(self):
        logs = list(patient_logs_collection.find())
        for log in logs:
            log['_id'] = str(log['_id'])  # Convert ObjectId to string
        return jsonify(logs)

    @jwt_required()
    def post(self):
        data = request.get_json()
        new_log = {
            'date': data['date'],
            'patient_id': data['patient_id'],
            'patient_name': data['patient_name'],
            'predicted_disease': data['predicted_disease'],
            'confidence': data['confidence']
        }
        result = patient_logs_collection.insert_one(new_log)
        new_log['_id'] = str(result.inserted_id)
        return jsonify({"message": "Log added successfully", "log": new_log}), 201

    @jwt_required()
    @admin_required
    def put(self, log_id):
        data = request.get_json()
        updated_log = {
            'date': data['date'],
            'patient_id': data['patient_id'],
            'patient_name': data['patient_name'],
            'predicted_disease': data['predicted_disease'],
            'confidence': data['confidence']
        }
        result = patient_logs_collection.update_one({'_id': ObjectId(log_id)}, {'$set': updated_log})
        if result.modified_count > 0:
            return jsonify({"message": "Log updated successfully"}), 200
        else:
            return jsonify({"message": "Log not found or no changes made"}), 404

class HeartRate(Resource):
    @jwt_required()
    def get(self):
        duration = int(request.args.get('duration', 60))  # Default to 60 seconds
        time_points, heart_rates = simulate_heart_rate(duration)
        return jsonify({'time': time_points.tolist(), 'heart_rate': heart_rates.tolist()})

api.add_resource(UserLogin, '/login')
api.add_resource(PatientLogs, '/patient_logs', '/patient_logs/<string:log_id>')
api.add_resource(HeartRate, '/heart_rate')

# Heart rate simulation
def simulate_heart_rate(duration):
    time_points = np.linspace(0, duration, num=duration*10)  # 10 samples per second
    base_rate = 70
    amplitude = 10
    frequency = 0.1
    noise = np.random.normal(0, 2, len(time_points))
    heart_rates = base_rate + amplitude * np.sin(2 * np.pi * frequency * time_points) + noise
    return time_points, heart_rates.astype(int)

if __name__ == '__main__':
    # Initialize the database with some sample data if it's empty
    if users_collection.count_documents({}) == 0:
        sample_users = [
            {'email': 'admin@example.com', 'password': 'admin123', 'role': 'admin'},
            {'email': 'doctor@example.com', 'password': 'doctor123', 'role': 'doctor'},
            {'email': 'user@example.com', 'password': 'user123', 'role': 'user'}
        ]
        users_collection.insert_many(sample_users)

    if patient_logs_collection.count_documents({}) == 0:
        sample_logs = [
            {'date': '2024-10-01 09:00:00', 'patient_id': 'P001', 'patient_name': 'John Doe', 'predicted_disease': 'Common Cold', 'confidence': '85%'},
            {'date': '2024-10-02 14:30:00', 'patient_id': 'P002', 'patient_name': 'Jane Smith', 'predicted_disease': 'Influenza', 'confidence': '92%'}
        ]
        patient_logs_collection.insert_many(sample_logs)

    app.run(debug=True,host="0.0.0.0")
