from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from sqlalchemy.exc import IntegrityError
from sqlalchemy import inspect
from werkzeug.security import generate_password_hash, check_password_hash
from apscheduler.schedulers.background import BackgroundScheduler
from joblib import load
import pandas as pd
from datetime import datetime, timedelta
import uuid
import json
import pytz
import re
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a strong secret key

# Configure the SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DB_URL", "sqlite:///resume.db")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Load the saved model and scaler
model_file_path = 'catboost.joblib'
gradient_boosting = load(model_file_path)
scaler = load('scaler.joblib')

# Define the features and scaler
# Define the original features as you had in your Jupyter Notebook
expected_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak',
                     'sex_0', 'sex_1', 'cp_0', 'cp_1', 'cp_2', 'cp_3',
                     'fbs_0', 'fbs_1', 'restecg_0', 'restecg_1', 'restecg_2',
                     'exang_0', 'exang_1', 'slope_0', 'slope_1', 'slope_2',
                     'ca_0', 'ca_1', 'ca_2', 'ca_3', 'ca_4',
                     'thal_0', 'thal_1', 'thal_2', 'thal_3']


# Database model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(10), unique=True, nullable=False)
    username = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    is_guest = db.Column(db.Boolean, nullable=False, default=False, server_default='0')
    session_id = db.Column(db.String(36), unique=True, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow,
                           server_default=db.text('CURRENT_TIMESTAMP'))

    __table_args__ = (
        db.UniqueConstraint('patient_id', name='uq_user_patient_id'),
        db.UniqueConstraint('email', name='uq_user_email'),
        db.UniqueConstraint('session_id', name='uq_user_session_id'),
    )

    def set_password(self, password):
        if not password and not self.is_guest:
            raise ValueError("Password is required for non-guest users")
        self.password_hash = generate_password_hash(password) if password else 'GUEST_USER_NO_LOGIN'

    def check_password(self, password):
        if self.is_guest:
            return False  # Guest users can't login with password
        return check_password_hash(self.password_hash, password)

    @staticmethod
    def cleanup_guest_users():
        """Remove guest users older than 1 hour"""
        expiry_time = datetime.utcnow() - timedelta(hours=1)
        guest_users = User.query.filter_by(is_guest=True).filter(User.created_at < expiry_time).all()

        for guest_user in guest_users:
            # Delete associated diagnoses
            DiagnosisRecord.query.filter_by(user_id=guest_user.id).delete()
            db.session.delete(guest_user)

        db.session.commit()


class DiagnosisRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    prediction = db.Column(db.Boolean, nullable=False)
    probability = db.Column(db.Float, nullable=False)
    form_data = db.Column(db.JSON, nullable=False)  # Add this line

    user = db.relationship('User', backref=db.backref('diagnoses', lazy=True))


# # Initialize the database
# with app.app_context():
#     db.create_all()


# Utility function to generate unique patient ID
def generate_patient_id():
    last_user = User.query.order_by(User.id.desc()).first()
    if last_user:
        return f"P{last_user.id + 1:05d}"
    else:
        return "P00001"


# Route for home page
@app.route('/')
def home():
    if 'user_id' in session:
        user = User.query.filter_by(id=session['user_id']).first()
        diagnoses = DiagnosisRecord.query.filter_by(user_id=user.id).order_by(DiagnosisRecord.timestamp.desc()).all()
        return render_template('home.html',
                               username=user.username,
                               patient_id=user.patient_id,
                               current_page='home',
                               diagnoses=diagnoses,
                               is_guest=user.is_guest)
    return redirect(url_for('login'))


# Add a scheduled task to clean up guest users
def setup_guest_cleanup():
    with app.app_context():
        User.cleanup_guest_users()


@app.route('/get_formatted_times', methods=['POST'])
def get_formatted_times():
    timezone_name = request.json['timezone']
    diagnosis_ids = request.json['diagnosis_ids']

    try:
        user_timezone = pytz.timezone(timezone_name)
    except pytz.exceptions.UnknownTimeZoneError:
        user_timezone = pytz.UTC

    formatted_times = {}
    for diagnosis_id in diagnosis_ids:
        diagnosis = DiagnosisRecord.query.get(diagnosis_id)
        if diagnosis:
            utc_time = diagnosis.timestamp.replace(tzinfo=pytz.UTC)
            local_time = utc_time.astimezone(user_timezone)
            formatted_times[diagnosis_id] = local_time.strftime('%d-%m-%Y %I:%M:%S %p')

    return jsonify(formatted_times)


def generate_guest_patient_id():
    return f"G{uuid.uuid4().hex[:8].upper()}"


@app.route('/guest-login', methods=['POST'])
def guest_login():
    # Cleanup old guest users
    User.cleanup_guest_users()

    # Create a guest user
    guest_username = f"Guest_{uuid.uuid4().hex[:6]}"
    guest_email = f"guest_{uuid.uuid4().hex[:8]}@temporary.com"
    guest_patient_id = generate_guest_patient_id()
    session_id = str(uuid.uuid4())

    guest_user = User(
        patient_id=guest_patient_id,
        username=guest_username,
        email=guest_email,
        is_guest=True,
        session_id=session_id
    )

    # Set a dummy password hash for guest users
    guest_user.password_hash = 'GUEST_USER_NO_LOGIN'

    try:
        db.session.add(guest_user)
        db.session.commit()
        session['user_id'] = guest_user.id
        session['is_guest'] = True
        flash('You are now logged in as a guest user. Your data will not be saved.', 'info')
        return redirect(url_for('home'))
    except Exception as e:
        db.session.rollback()
        flash('An error occurred while creating guest session.', 'error')
        return redirect(url_for('login'))


# Route for registration
# Modify the register route to ensure password is always required
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        email = request.form['email'].strip()
        password = request.form['password'].strip()

        # Check for empty password
        if not password:
            flash('Password is required!', 'error')
            return render_template('register.html')

        # Basic email validation
        email_regex = r'^\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,4}\b'
        if not re.match(email_regex, email):
            flash('Invalid email address!', 'error')
            return render_template('register.html')

        # Generate unique patient ID
        patient_id = generate_patient_id()

        # Create new user
        new_user = User(
            patient_id=patient_id,
            username=username,
            email=email,
            is_guest=False
        )
        new_user.set_password(password)

        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except IntegrityError:
            db.session.rollback()
            flash('Email already exists!', 'error')
            return render_template('register.html')

    return render_template('register.html')


# Modify the login route to prevent guest user login attempts
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        identifier = request.form['identifier'].strip()
        password = request.form['password'].strip()

        # Determine if identifier is email or patient_id
        if re.match(r'^\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,4}\b', identifier):
            user = User.query.filter_by(email=identifier).first()
        else:
            user = User.query.filter_by(patient_id=identifier).first()

        if user and user.is_guest:
            flash('Guest accounts cannot be used for login. Please register for a full account.', 'error')
            return render_template('login.html')

        if user and user.check_password(password):
            session['user_id'] = user.id
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials!', 'error')
            return render_template('login.html')

    return render_template('login.html')


# Route for logout
@app.route('/logout')
def logout():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user and user.is_guest:
            # Delete guest user and their data
            DiagnosisRecord.query.filter_by(user_id=user.id).delete()
            db.session.delete(user)
            db.session.commit()

    session.pop('user_id', None)
    session.pop('is_guest', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))


# Route for diagnosis
@app.route('/diagnose', methods=['GET', 'POST'])
def diagnose():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        # Get input data from form
        user_input = {
            'age': int(request.form['age']),
            'sex': int(request.form['sex']),
            'cp': int(request.form['cp']),
            'trestbps': int(request.form['trestbps']),
            'chol': int(request.form['chol']),
            'fbs': int(request.form['fbs']),
            'restecg': int(request.form['restecg']),
            'thalach': int(request.form['thalach']),
            'exang': int(request.form['exang']),
            'oldpeak': float(request.form['oldpeak']),
            'slope': int(request.form['slope']),
            'ca': int(request.form['ca']),
            'thal': int(request.form['thal'])
        }

        # Convert user input to DataFrame
        input_df = pd.DataFrame(user_input, index=[0])

        # One-hot encode the categorical variables
        categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        input_df_encoded = pd.get_dummies(input_df, columns=categorical_columns, prefix=categorical_columns)

        # Ensure all expected columns are present
        for column in expected_features:
            if column not in input_df_encoded.columns:
                input_df_encoded[column] = 0

        # Reorder columns to match the expected order
        input_df_encoded = input_df_encoded[expected_features]

        # Scale the user input
        scaled_input = scaler.transform(input_df_encoded)

        # Make prediction
        user_prediction = gradient_boosting.predict(scaled_input)
        user_prediction_proba = gradient_boosting.predict_proba(scaled_input)

        # Display the result as a percentage
        prediction_percentage = user_prediction_proba[0][1] * 100

        new_record = DiagnosisRecord(
            user_id=session['user_id'],
            prediction=bool(user_prediction[0]),
            probability=prediction_percentage,
            form_data=user_input
        )
        db.session.add(new_record)
        db.session.commit()

        result = {
            "prediction": user_prediction[0],
            "probability": prediction_percentage
        }

        return render_template('result.html', result=result, form_data=user_input, current_page='result')

    return render_template('diagnose.html', current_page='diagnose')


@app.route('/delete_diagnosis/<int:diagnosis_id>', methods=['POST'])
def delete_diagnosis(diagnosis_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    diagnosis = DiagnosisRecord.query.get_or_404(diagnosis_id)

    # Ensure the logged-in user owns this diagnosis
    if diagnosis.user_id != session['user_id']:
        flash('You do not have permission to delete this diagnosis.', 'error')
        return redirect(url_for('home'))

    db.session.delete(diagnosis)
    db.session.commit()

    flash('Diagnosis record deleted successfully.', 'success')
    return redirect(url_for('home'))


@app.route('/view_diagnosis/<int:diagnosis_id>')
def view_diagnosis(diagnosis_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    diagnosis = DiagnosisRecord.query.get_or_404(diagnosis_id)

    # Ensure the logged-in user owns this diagnosis
    if diagnosis.user_id != session['user_id']:
        flash('You do not have permission to view this diagnosis.', 'error')
        return redirect(url_for('home'))

    result = {
        "prediction": diagnosis.prediction,
        "probability": diagnosis.probability
    }

    # Retrieve the form data
    form_data = diagnosis.form_data

    # If form_data is stored as a string, parse it
    if isinstance(form_data, str):
        form_data = json.loads(form_data)

    return render_template('result.html', result=result, form_data=form_data, current_page='result')


with app.app_context():
    if not inspect(db.engine).get_table_names():
        db.create_all()
        scheduler = BackgroundScheduler()
        scheduler.add_job(setup_guest_cleanup, 'interval', hours=1)
        scheduler.start()

# if __name__ == '__main__':
#     app.run(debug=True)
