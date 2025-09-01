# database.py
from models import db


def init_db(app):
    """Initialize the database with the Flask app"""
    db.init_app(app)

    with app.app_context():
        # Create all tables
        db.create_all()
        print("✅ Database tables created successfully!")