# init_database.py
from flask import Flask
from models import db
from models import Tourist, Location, Alert

# Create a simple Flask app just for initialization
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tourist_safety.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize db with this app
db.init_app(app)

print("🚀 Creating database tables for localhost...")

with app.app_context():
    # Drop all existing tables (if any)
    db.drop_all()

    # Create all tables
    db.create_all()

    print("✅ Database tables created successfully!")
    print("📊 Tables created: tourists, locations, alerts")
    print("🌐 Now run: python app.py")
    print("🔗 Open: http://localhost:5000")