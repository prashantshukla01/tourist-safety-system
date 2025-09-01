import os
import joblib
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from geopy.distance import geodesic
import uuid

# Import your modules
from models import db, Tourist, Location, Alert
# ADD THIS RIGHT AFTER IMPORTS IN app.py
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class EnsembleAnomalyDetector(BaseEstimator, TransformerMixin):
    """Ensemble of Isolation Forest and Local Outlier Factor for anomaly detection"""

    def __init__(self, contamination=0.1, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.lof = LocalOutlierFactor(
            contamination=contamination,
            novelty=True,
            n_neighbors=20
        )
        self.weights = [0.6, 0.4]  # Weight for ISO Forest and LOF respectively

    def fit(self, X, y=None):
        self.iso_forest.fit(X)
        self.lof.fit(X)
        return self

    def predict(self, X):
        iso_scores = self.iso_forest.decision_function(X)
        lof_scores = self.lof.decision_function(X)

        # Normalize scores to [0, 1] range
        iso_scores_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
        lof_scores_norm = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min())

        # Weighted ensemble score
        ensemble_scores = self.weights[0] * iso_scores_norm + self.weights[1] * lof_scores_norm

        # Convert to binary predictions (1 for normal, -1 for anomaly)
        threshold = np.percentile(ensemble_scores, 100 * self.contamination)
        predictions = np.where(ensemble_scores < threshold, -1, 1)

        return predictions, ensemble_scores

    def decision_function(self, X):
        _, scores = self.predict(X)
        return scores

# Flask app initialization - simple localhost setup
app = Flask(__name__)

# Simple configuration for localhost
app.config['DEBUG'] = True
app.config['SECRET_KEY'] = 'simple-dev-key-123'
app.config['JWT_SECRET_KEY'] = 'jwt-simple-key-456'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tourist_safety.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Risk zones (latitude, longitude, radius in meters)
app.config['RISK_ZONES'] = [
    ((40.7589, -73.9851), 500),  # Times Square example
    ((40.7484, -73.9857), 300),  # Empire State Building example
]

# Initialize extensions
jwt = JWTManager(app)
db.init_app(app)

# Initialize database tables
with app.app_context():
    db.create_all()
    print("✅ Database tables created successfully!")

# Load ML model and scaler
MODEL_PATH = os.path.join('models', 'anomaly_detection_model.joblib')
SCALER_PATH = os.path.join('models', 'scaler.joblib')

model, scaler = None, None
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ ML Model & Scaler loaded successfully.")
except Exception as e:
    print("❌ Error loading ML model:", e)

# Helper functions
def calculate_velocity(prev_lat, prev_lon, prev_time, curr_lat, curr_lon, curr_time):
    """Calculate velocity between two points in m/s"""
    if not all([prev_lat, prev_lon, prev_time, curr_lat, curr_lon, curr_time]):
        return 0.0

    try:
        # Calculate distance in meters
        prev_point = (prev_lat, prev_lon)
        curr_point = (curr_lat, curr_lon)
        distance = geodesic(prev_point, curr_point).meters

        # Calculate time difference in seconds
        time_diff = (curr_time - prev_time).total_seconds()

        if time_diff > 0:
            return distance / time_diff
        else:
            return 0.0
    except:
        return 0.0

def calculate_acceleration(prev_velocity, curr_velocity, time_diff):
    """Calculate acceleration in m/s²"""
    if time_diff > 0:
        return (curr_velocity - prev_velocity) / time_diff
    return 0.0

def calculate_distance_from_risk_zones(lat, lon, risk_zones):
    """Calculate minimum distance from any risk zone"""
    if not risk_zones:
        return float('inf')

    current_point = (lat, lon)
    min_distance = float('inf')

    for zone_center, zone_radius in risk_zones:
        distance = geodesic(current_point, zone_center).meters
        if distance < min_distance:
            min_distance = distance

    return min_distance

def detect_anomaly(lat, lon, velocity, acceleration, timestamp):
    """Use ML model to detect anomalies"""
    if model is None or scaler is None:
        return 0.0, False

    try:
        # Create feature vector
        hour_of_day = timestamp.hour
        day_of_week = timestamp.weekday()

        # Calculate distance from nearest risk zone
        risk_zone_distance = calculate_distance_from_risk_zones(lat, lon, app.config['RISK_ZONES'])

        features = np.array([[velocity, acceleration, risk_zone_distance, hour_of_day, day_of_week]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Get prediction
        prediction, score = model.named_steps['detector'].predict(features_scaled)

        # Convert score to anomaly probability (0-1)
        anomaly_prob = 1 - (score[0] + 1) / 2  # Convert from [-1, 1] to [0, 1]

        is_anomaly = prediction[0] == -1

        return anomaly_prob, is_anomaly
    except Exception as e:
        print(f"Error in anomaly detection: {e}")
        return 0.0, False

# Routes
@app.route('/')
def index():
    """Serve the dashboard"""
    return render_template('dashboard.html')

@app.route('/login')
def login_page():
    """Serve the login page"""
    return render_template('login.html')

@app.route('/register')
def register_page():
    """Serve the register page"""
    return render_template('register.html')

@app.route('/alerts')
def alerts_page():
    """Serve the alerts page"""
    return render_template('alerts.html')

@app.route('/api/register', methods=['POST'])
def register():
    """Register a new tourist"""
    try:
        data = request.get_json()
        print(f"Registration data: {data}")  # Debug log

        # Validate required fields
        required_fields = ['username', 'email', 'password', 'first_name', 'last_name']
        for field in required_fields:
            if field not in data:
                print(f"Missing field: {field}")  # Debug log
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Check if user already exists
        if Tourist.query.filter_by(username=data['username']).first():
            print("Username already exists")  # Debug log
            return jsonify({'error': 'Username already exists'}), 400

        if Tourist.query.filter_by(email=data['email']).first():
            print("Email already exists")  # Debug log
            return jsonify({'error': 'Email already exists'}), 400

        # Create new tourist
        tourist = Tourist(
            username=data['username'],
            email=data['email'],
            password_hash=generate_password_hash(data['password']),
            first_name=data['first_name'],
            last_name=data['last_name'],
            phone=data.get('phone', ''),
            emergency_contact=data.get('emergency_contact', '')
        )

        db.session.add(tourist)
        db.session.commit()
        print(f"User created: {tourist.username}")  # Debug log

        # Generate access token
        access_token = create_access_token(identity=tourist.id)

        return jsonify({
            'message': 'User created successfully',
            'access_token': access_token,
            'user': tourist.to_dict()
        }), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    """Login a tourist"""
    try:
        data = request.get_json()

        # Validate required fields
        if 'username' not in data or 'password' not in data:
            return jsonify({'error': 'Username and password are required'}), 400

        # Find user
        tourist = Tourist.query.filter_by(username=data['username']).first()
        if not tourist or not check_password_hash(tourist.password_hash, data['password']):
            return jsonify({'error': 'Invalid username or password'}), 401

        # Generate access token
        access_token = create_access_token(identity=tourist.id)

        return jsonify({
            'message': 'Login successful',
            'access_token': access_token,
            'user': tourist.to_dict()
        }), 200


    except Exception as e:

        print(f"Registration error: {str(e)}")  # Debug log

        return jsonify({'error': str(e)}), 500

@app.route('/api/location', methods=['POST'])
@jwt_required()
def update_location():
    """Update tourist location and get anomaly prediction"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()

        # Validate required fields
        if 'latitude' not in data or 'longitude' not in data:
            return jsonify({'error': 'Latitude and longitude are required'}), 400

        lat = float(data['latitude'])
        lon = float(data['longitude'])
        timestamp = datetime.utcnow()

        # Get tourist's previous location
        prev_location = Location.query.filter_by(tourist_id=current_user_id) \
            .order_by(Location.timestamp.desc()).first()

        # Calculate velocity and acceleration
        if prev_location:
            velocity = calculate_velocity(
                prev_location.latitude, prev_location.longitude, prev_location.timestamp,
                lat, lon, timestamp
            )
            time_diff = (timestamp - prev_location.timestamp).total_seconds()
            acceleration = calculate_acceleration(prev_location.velocity, velocity, time_diff)
        else:
            velocity = 0.0
            acceleration = 0.0

        # Detect anomalies using ML model
        anomaly_score, is_anomaly = detect_anomaly(lat, lon, velocity, acceleration, timestamp)

        # Check if in risk zone
        in_risk_zone = False
        for zone_center, zone_radius in app.config['RISK_ZONES']:
            distance = geodesic((lat, lon), zone_center).meters
            if distance <= zone_radius:
                in_risk_zone = True
                break

        # Create alert if anomaly detected or in risk zone
        if is_anomaly or in_risk_zone:
            alert_type = 'risk_zone' if in_risk_zone else 'anomaly'
            severity = 'high' if in_risk_zone else ('medium' if anomaly_score > 0.7 else 'low')

            alert = Alert(
                tourist_id=current_user_id,
                alert_type=alert_type,
                severity=severity,
                latitude=lat,
                longitude=lon,
                description=f"{'Risk zone entry' if in_risk_zone else 'Anomalous behavior'} detected with score {anomaly_score:.2f}"
            )
            db.session.add(alert)

        # Save location
        location = Location(
            tourist_id=current_user_id,
            latitude=lat,
            longitude=lon,
            timestamp=timestamp,
            velocity=velocity,
            acceleration=acceleration,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly
        )
        db.session.add(location)
        db.session.commit()

        return jsonify({
            'message': 'Location updated successfully',
            'velocity': velocity,
            'acceleration': acceleration,
            'anomaly_score': anomaly_score,
            'is_anomaly': is_anomaly,
            'in_risk_zone': in_risk_zone,
            'alert_triggered': is_anomaly or in_risk_zone
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/panic', methods=['POST'])
@jwt_required()
def panic_button():
    """Handle panic button press"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()

        # Get current location if provided, otherwise use last known location
        if data and 'latitude' in data and 'longitude' in data:
            lat = float(data['latitude'])
            lon = float(data['longitude'])
        else:
            last_location = Location.query.filter_by(tourist_id=current_user_id) \
                .order_by(Location.timestamp.desc()).first()
            if last_location:
                lat = last_location.latitude
                lon = last_location.longitude
            else:
                return jsonify({'error': 'No location data available'}), 400

        # Create high severity alert
        alert = Alert(
            tourist_id=current_user_id,
            alert_type='panic',
            severity='high',
            latitude=lat,
            longitude=lon,
            description='Panic button activated by user'
        )
        db.session.add(alert)
        db.session.commit()

        return jsonify({
            'message': 'Panic alert created successfully',
            'alert': alert.to_dict()
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts', methods=['GET'])
@jwt_required()
def get_alerts():
    """Get alert history for the authenticated tourist"""
    try:
        current_user_id = get_jwt_identity()

        # Get query parameters
        limit = request.args.get('limit', 50, type=int)
        resolved = request.args.get('resolved', None, type=str)

        # Build query
        query = Alert.query.filter_by(tourist_id=current_user_id)

        if resolved is not None:
            if resolved.lower() == 'true':
                query = query.filter_by(is_resolved=True)
            else:
                query = query.filter_by(is_resolved=False)

        alerts = query.order_by(Alert.timestamp.desc()).limit(limit).all()

        return jsonify({
            'alerts': [alert.to_dict() for alert in alerts]
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
@jwt_required()
def get_location_history():
    """Get location history for the authenticated tourist"""
    try:
        current_user_id = get_jwt_identity()

        # Get query parameters
        limit = request.args.get('limit', 100, type=int)
        hours = request.args.get('hours', 24, type=int)

        # Calculate time threshold
        time_threshold = datetime.utcnow() - timedelta(hours=hours)

        # Query locations
        locations = Location.query.filter(
            Location.tourist_id == current_user_id,
            Location.timestamp >= time_threshold
        ).order_by(Location.timestamp.desc()).limit(limit).all()

        return jsonify({
            'locations': [loc.to_dict() for loc in locations]
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@jwt.unauthorized_loader
def unauthorized_handler(callback):
    return jsonify({'error': 'Missing or invalid token'}), 401

@jwt.invalid_token_loader
def invalid_token_handler(callback):
    return jsonify({'error': 'Invalid token'}), 401

@jwt.expired_token_loader
def expired_token_handler(callback):
    return jsonify({'error': 'Token has expired'}), 401

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)