from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import uuid

# Create db instance only once
db = SQLAlchemy()

class Tourist(db.Model):
    __tablename__ = 'tourists'

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    first_name = db.Column(db.String(80), nullable=False)
    last_name = db.Column(db.String(80), nullable=False)
    phone = db.Column(db.String(20))
    emergency_contact = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)

    locations = db.relationship('Location', backref='tourist', lazy=True)
    alerts = db.relationship('Alert', backref='tourist', lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'phone': self.phone,
            'emergency_contact': self.emergency_contact,
            'created_at': self.created_at.isoformat()
        }


class Location(db.Model):
    __tablename__ = 'locations'

    id = db.Column(db.Integer, primary_key=True)
    tourist_id = db.Column(db.String(36), db.ForeignKey('tourists.id'), nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    velocity = db.Column(db.Float, default=0.0)
    acceleration = db.Column(db.Float, default=0.0)
    anomaly_score = db.Column(db.Float, default=0.0)
    is_anomaly = db.Column(db.Boolean, default=False)

    def to_dict(self):
        return {
            'id': self.id,
            'tourist_id': self.tourist_id,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'timestamp': self.timestamp.isoformat(),
            'velocity': self.velocity,
            'acceleration': self.acceleration,
            'anomaly_score': self.anomaly_score,
            'is_anomaly': self.is_anomaly
        }


class Alert(db.Model):
    __tablename__ = 'alerts'

    id = db.Column(db.Integer, primary_key=True)
    tourist_id = db.Column(db.String(36), db.ForeignKey('tourists.id'), nullable=False)
    alert_type = db.Column(db.String(20), nullable=False)
    severity = db.Column(db.String(10), nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    description = db.Column(db.Text)
    is_resolved = db.Column(db.Boolean, default=False)
    resolved_at = db.Column(db.DateTime)

    def to_dict(self):
        return {
            'id': self.id,
            'tourist_id': self.tourist_id,
            'alert_type': self.alert_type,
            'severity': self.severity,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'timestamp': self.timestamp.isoformat(),
            'description': self.description,
            'is_resolved': self.is_resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }