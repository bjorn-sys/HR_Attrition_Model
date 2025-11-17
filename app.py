# --------------------------------------------------------------------
# 👥 HR Employee Attrition Analytics App - SQLite Database Version
# --------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import uuid
import base64
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import HorizontalBarChart
import sqlite3
import json
import os
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# DATABASE SETUP
# =============================================================================
DB_FILE = 'hr_attrition_database.db'

def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Employees table
    c.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            employee_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER,
            department TEXT,
            position TEXT,
            salary REAL,
            contact TEXT,
            hire_date TEXT,
            notes TEXT,
            created_date TEXT,
            updated_date TEXT
        )
    ''')
    
    # Predictions table
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT,
            timestamp TEXT,
            prediction TEXT,
            probability REAL,
            risk_level TEXT,
            threshold REAL,
            features TEXT,
            FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
        )
    ''')
    
    # HR settings table
    c.execute('''
        CREATE TABLE IF NOT EXISTS hr_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            retention_notes_text TEXT,
            development_plan_text TEXT,
            risk_factors_text TEXT,
            created_date TEXT
        )
    ''')
    
    # Create default settings if none exist
    c.execute('SELECT COUNT(*) FROM hr_settings')
    if c.fetchone()[0] == 0:
        default_retention = "Schedule one-on-one meeting\nReview career development path\nAssess compensation package\nMonitor work-life balance"
        default_development = "Skills training programs\nLeadership development\nCross-functional projects\nMentorship opportunities"
        default_risk_factors = "Low job satisfaction\nFrequent overtime\nPoor work-life balance\nLimited growth opportunities"
        
        c.execute('''
            INSERT INTO hr_settings (retention_notes_text, development_plan_text, risk_factors_text, created_date)
            VALUES (?, ?, ?, ?)
        ''', (default_retention, default_development, default_risk_factors, datetime.now().strftime("%Y-%m-%d %H:%M")))
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_database()

# =============================================================================
# DATABASE OPERATIONS
# =============================================================================
def save_employee(employee_data):
    """Save or update employee in database"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Check if employee exists
    c.execute('SELECT COUNT(*) FROM employees WHERE employee_id = ?', (employee_data['employee_id'],))
    exists = c.fetchone()[0] > 0
    
    if exists:
        # Update existing employee
        c.execute('''
            UPDATE employees 
            SET name=?, age=?, department=?, position=?, salary=?, contact=?, hire_date=?, notes=?, updated_date=?
            WHERE employee_id=?
        ''', (
            employee_data['name'], employee_data['age'], employee_data['department'],
            employee_data['position'], employee_data['salary'], employee_data['contact'],
            employee_data['hire_date'], employee_data['notes'],
            datetime.now().strftime("%Y-%m-%d %H:%M"), employee_data['employee_id']
        ))
    else:
        # Insert new employee
        c.execute('''
            INSERT INTO employees 
            (employee_id, name, age, department, position, salary, contact, hire_date, notes, created_date, updated_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            employee_data['employee_id'], employee_data['name'], employee_data['age'],
            employee_data['department'], employee_data['position'], employee_data['salary'],
            employee_data['contact'], employee_data['hire_date'], employee_data['notes'],
            employee_data['created_date'], datetime.now().strftime("%Y-%m-%d %H:%M")
        ))
    
    conn.commit()
    conn.close()

def save_prediction(prediction_data):
    """Save prediction to database"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO predictions 
        (employee_id, timestamp, prediction, probability, risk_level, threshold, features)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        prediction_data['employee_id'],
        prediction_data['timestamp'],
        prediction_data['prediction'],
        prediction_data['probability'],
        prediction_data['risk_level'],
        prediction_data['threshold'],
        json.dumps(prediction_data['features'])
    ))
    
    conn.commit()
    conn.close()

def get_employee(employee_id):
    """Get single employee by ID"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('SELECT * FROM employees WHERE employee_id = ?', (employee_id,))
    row = c.fetchone()
    
    if row:
        employee = {
            'employee_id': row[0],
            'name': row[1],
            'age': row[2],
            'department': row[3],
            'position': row[4],
            'salary': row[5],
            'contact': row[6],
            'hire_date': row[7],
            'notes': row[8],
            'created_date': row[9],
            'updated_date': row[10]
        }
        conn.close()
        return employee
    conn.close()
    return None

def search_employees(search_term="", limit=50, offset=0):
    """Search employees with pagination"""
    conn = sqlite3.connect(DB_FILE)
    
    if search_term:
        query = '''
            SELECT * FROM employees 
            WHERE name LIKE ? OR employee_id LIKE ? OR department LIKE ? OR position LIKE ?
            ORDER BY updated_date DESC 
            LIMIT ? OFFSET ?
        '''
        search_pattern = f'%{search_term}%'
        employees_df = pd.read_sql_query(query, conn, params=(search_pattern, search_pattern, search_pattern, search_pattern, limit, offset))
    else:
        query = 'SELECT * FROM employees ORDER BY updated_date DESC LIMIT ? OFFSET ?'
        employees_df = pd.read_sql_query(query, conn, params=(limit, offset))
    
    conn.close()
    return employees_df

def get_employee_count():
    """Get total number of employees"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM employees')
    count = c.fetchone()[0]
    conn.close()
    return count

def get_employee_predictions(employee_id, limit=5):
    """Get recent predictions for an employee"""
    conn = sqlite3.connect(DB_FILE)
    
    query = '''
        SELECT * FROM predictions 
        WHERE employee_id = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
    '''
    predictions_df = pd.read_sql_query(query, conn, params=(employee_id, limit))
    
    # Parse features JSON
    if not predictions_df.empty:
        predictions_df['features'] = predictions_df['features'].apply(lambda x: json.loads(x) if x else {})
    
    conn.close()
    return predictions_df

def get_hr_settings():
    """Get HR settings"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('SELECT retention_notes_text, development_plan_text, risk_factors_text FROM hr_settings ORDER BY id DESC LIMIT 1')
    row = c.fetchone()
    
    conn.close()
    
    if row:
        return {
            'retention_notes_text': row[0],
            'development_plan_text': row[1],
            'risk_factors_text': row[2]
        }
    return None

def save_hr_settings(settings):
    """Save HR settings"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO hr_settings (retention_notes_text, development_plan_text, risk_factors_text, created_date)
        VALUES (?, ?, ?, ?)
    ''', (
        settings['retention_notes_text'],
        settings['development_plan_text'],
        settings['risk_factors_text'],
        datetime.now().strftime("%Y-%m-%d %H:%M")
    ))
    
    conn.commit()
    conn.close()

# =============================================================================
# AUTHENTICATION SYSTEM
# =============================================================================
def check_login(username, password):
    """Check if username and password are correct"""
    return username == 'admin' and password == 'admin123'

def login_section():
    """Display login form and handle authentication"""
    if not st.session_state.get('logged_in', False):
        st.sidebar.header("🔐 HR Admin Login")
        
        with st.sidebar.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            login_button = st.form_submit_button("Login")
            
            if login_button:
                if check_login(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.sidebar.success(f"Welcome, {username}!")
                    st.rerun()
                else:
                    st.sidebar.error("Invalid username or password")
        
        return False
    else:
        st.sidebar.success(f"✅ Logged in as: {st.session_state.username}")
        if st.sidebar.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()
        return True

# =============================================================================
# DATA EXPORT FUNCTIONS
# =============================================================================
def export_all_hr_records():
    """Export all employee records with their predictions as a comprehensive CSV"""
    conn = sqlite3.connect(DB_FILE)
    
    # Get all employees
    employees_df = pd.read_sql_query("SELECT * FROM employees", conn)
    
    # Get all predictions
    predictions_df = pd.read_sql_query("SELECT * FROM predictions", conn)
    
    conn.close()
    
    if employees_df.empty:
        return None
    
    # Create comprehensive HR records
    hr_records = []
    
    for _, employee in employees_df.iterrows():
        # Get employee's predictions
        employee_predictions = predictions_df[predictions_df['employee_id'] == employee['employee_id']]
        
        if not employee_predictions.empty:
            # Create one row per prediction
            for _, prediction in employee_predictions.iterrows():
                # Parse features JSON
                features = {}
                if prediction['features']:
                    try:
                        features = json.loads(prediction['features'])
                    except:
                        features = {}
                
                record = {
                    'employee_id': employee['employee_id'],
                    'employee_name': employee['name'],
                    'age': employee['age'],
                    'department': employee['department'],
                    'position': employee['position'],
                    'salary': f"${employee['salary']:,.2f}",
                    'contact': employee['contact'],
                    'hire_date': employee['hire_date'],
                    'employee_notes': employee['notes'],
                    'prediction_id': prediction['id'],
                    'prediction_date': prediction['timestamp'],
                    'attrition_prediction': prediction['prediction'],
                    'attrition_probability': f"{prediction['probability']:.2f}%",
                    'risk_level': prediction['risk_level'],
                    'threshold_used': prediction['threshold']
                }
                
                # Add all HR features
                for feature, value in features.items():
                    feature_name = feature.replace('_', ' ').title()
                    record[feature_name] = f"{value:.4f}"
                
                hr_records.append(record)
        else:
            # Employee with no predictions
            record = {
                'employee_id': employee['employee_id'],
                'employee_name': employee['name'],
                'age': employee['age'],
                'department': employee['department'],
                'position': employee['position'],
                'salary': f"${employee['salary']:,.2f}",
                'contact': employee['contact'],
                'hire_date': employee['hire_date'],
                'employee_notes': employee['notes'],
                'prediction_id': 'No prediction',
                'prediction_date': 'No prediction',
                'attrition_prediction': 'No prediction',
                'attrition_probability': 'N/A',
                'risk_level': 'N/A',
                'threshold_used': 'N/A'
            }
            hr_records.append(record)
    
    return pd.DataFrame(hr_records)

def download_all_data_section():
    """Section for downloading all HR records"""
    if st.session_state.get('logged_in', False):
        st.sidebar.header("📊 HR Data Export")
        
        st.sidebar.warning("⚠️ Export your data regularly! Database may reset on redeployment.")
        
        # Export all HR records
        if st.sidebar.button("📥 Download All HR Records (CSV)"):
            with st.spinner("Generating comprehensive HR records export..."):
                hr_records_df = export_all_hr_records()
                
                if hr_records_df is not None and not hr_records_df.empty:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    csv = hr_records_df.to_csv(index=False)
                    
                    st.sidebar.download_button(
                        label="💾 Download HR Records CSV",
                        data=csv,
                        file_name=f"hr_attrition_records_{timestamp}.csv",
                        mime="text/csv",
                        key="download_all_hr_records"
                    )
                    
                    # Show preview
                    st.sidebar.info(f"✅ Ready to download {len(hr_records_df)} records")
                    
                    with st.sidebar.expander("📋 Preview HR Records"):
                        st.dataframe(hr_records_df.head(3))
                else:
                    st.sidebar.error("No HR records found to export")
        
        # Quick stats
        if st.sidebar.button("📈 Show Database Statistics"):
            conn = sqlite3.connect(DB_FILE)
            employee_count = get_employee_count()
            prediction_count = pd.read_sql_query("SELECT COUNT(*) as count FROM predictions", conn)['count'][0]
            conn.close()
            
            st.sidebar.success(f"""
            **Database Statistics:**
            - Employees: {employee_count}
            - Predictions: {prediction_count}
            - Total Records: {employee_count + prediction_count}
            """)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="HR Attrition Analytics Pro",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
def initialize_session_state():
    """Initialize session state with database values"""
    defaults = {
        'current_employee_id': None,
        'single_pred': None,
        'batch_pred': None,
        'show_tutorial': True,
        'show_new_employee_form': False,
        'inputs': {
            'job_satisfaction': 3.0,
            'environment_satisfaction': 3.0,
            'work_life_balance': 3.0,
            'job_involvement': 3.0,
            'years_at_company': 3.0,
            'monthly_income': 5000.0,
            'overtime': 0.0,
            'age': 35.0,
            'years_since_last_promotion': 2.0,
            'training_times_last_year': 2.0
        },
        'current_page': 0,
        'employees_per_page': 20,
        'search_term': "",
        'logged_in': False,
        'username': None
    }
    
    # Set defaults
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Load HR settings from database
    settings = get_hr_settings()
    if settings:
        for key, value in settings.items():
            if key not in st.session_state:
                st.session_state[key] = value

initialize_session_state()

# =============================================================================
# PREDICTION LOGIC
# =============================================================================
class AttritionPredictor:
    def __init__(self):
        self.features = [
            'job_satisfaction', 'environment_satisfaction', 'work_life_balance',
            'job_involvement', 'years_at_company', 'monthly_income', 'overtime',
            'age', 'years_since_last_promotion', 'training_times_last_year'
        ]

    def predict(self, df, threshold=0.5):
        """
        Simple rule-based prediction logic for employee attrition.
        """
        try:
            # Ensure all required features are present
            for feature in self.features:
                if feature not in df.columns:
                    df[feature] = 0.0

            probs = []
            for _, row in df.iterrows():
                # Simple scoring system based on feature values
                score = 0
                
                # Risk factors (higher values increase risk)
                if row['job_satisfaction'] < 2.5: score += 2
                if row['environment_satisfaction'] < 2.5: score += 2
                if row['work_life_balance'] < 2.0: score += 2
                if row['overtime'] > 0.5: score += 1
                if row['years_since_last_promotion'] > 3: score += 1
                if row['training_times_last_year'] < 2: score += 1
                
                # Protective factors (higher values decrease risk)
                if row['job_involvement'] > 3.0: score -= 1
                if row['years_at_company'] > 5: score -= 1
                if row['monthly_income'] > 8000: score -= 1
                if row['age'] > 40: score -= 1

                # Convert score to probability (0 to 0.9 range)
                score = max(0, min(8, score))  # Cap score between 0-8
                prob_attrition = min(0.9, score * 0.1125)  # Scale to 0-0.9
                prob_stay = 1 - prob_attrition
                probs.append([prob_stay, prob_attrition])

            probs = np.array(probs)
            preds = (probs[:, 1] >= threshold).astype(int)

            # Assign risk levels based on probability
            risks = []
            for prob in probs[:, 1]:
                if prob < 0.2: 
                    risks.append(("Low Risk", prob))
                elif prob < 0.4: 
                    risks.append(("Moderate Risk", prob))
                elif prob < 0.61: 
                    risks.append(("High Risk", prob))
                else: 
                    risks.append(("Critical Risk", prob))

            return preds, probs, risks

        except Exception as e:
            st.error(f"Prediction error: {e}")
            # Return safe default values
            n = len(df)
            return (
                np.zeros(n), 
                np.array([[0.8, 0.2]] * n), 
                [("Low Risk", 0.2)] * n
            )

# =============================================================================
# PDF REPORT GENERATION
# =============================================================================
def generate_pdf_report(employee_data, prediction_data, input_data, retention_notes, development_plan, risk_factors):
    """Generate a PDF report for the employee analysis."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        alignment=1,
        textColor=colors.HexColor('#2E86AB')
    )
    story.append(Paragraph("EMPLOYEE ATTRITION RISK ANALYSIS REPORT", title_style))
    story.append(Spacer(1, 20))
    
    # Employee Information Section
    story.append(Paragraph("EMPLOYEE INFORMATION", styles['Heading2']))
    employee_info = [
        ["Employee Name:", employee_data['name']],
        ["Employee ID:", employee_data['employee_id']],
        ["Age:", str(employee_data['age'])],
        ["Department:", employee_data['department']],
        ["Position:", employee_data['position']],
        ["Salary:", f"${employee_data.get('salary', 0):,.2f}"],
        ["Contact:", employee_data.get('contact', 'N/A')],
        ["Hire Date:", employee_data.get('hire_date', 'N/A')],
        ["Report Date:", datetime.now().strftime("%Y-%m-%d %H:%M")]
    ]
    
    employee_table = Table(employee_info, colWidths=[2*inch, 3*inch])
    employee_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F8F9FA')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#FFFFFF')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#CCCCCC'))
    ]))
    story.append(employee_table)
    story.append(Spacer(1, 20))
    
    # Prediction Results Section
    story.append(Paragraph("ATTRITION RISK ANALYSIS", styles['Heading2']))
    
    preds, probs, risks = prediction_data['pred'], prediction_data['probs'], prediction_data['risks']
    prediction_label = "High Attrition Risk" if preds[0] == 1 else "Low Attrition Risk"
    risk_level, probability = risks[0][0], probs[0][1]
    
    results_info = [
        ["Prediction:", f"<b>{prediction_label}</b>"],
        ["Risk Level:", f"<b>{risk_level}</b>"],
        ["Attrition Probability:", f"<b>{probability:.1%}</b>"],
        ["Confidence Score:", f"<b>{max(probs[0])*100:.1f}%</b>"],
        ["Analysis Date:", datetime.now().strftime("%Y-%m-%d %H:%M")]
    ]
    
    results_table = Table(results_info, colWidths=[2*inch, 3*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FFFFFF')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#CCCCCC'))
    ]))
    story.append(results_table)
    story.append(Spacer(1, 20))
    
    # Employee Metrics Section
    story.append(Paragraph("EMPLOYEE METRICS", styles['Heading2']))
    metrics_data = [["Metric", "Value"]]
    for metric, value in input_data.items():
        display_name = metric.replace('_', ' ').title()
        if metric == 'monthly_income':
            metrics_data.append([display_name, f"${value:,.2f}"])
        else:
            metrics_data.append([display_name, f"{value:.2f}"])
    
    metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#CCCCCC'))
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 20))
    
    # Retention Strategies Section
    if retention_notes and retention_notes.strip():
        story.append(Paragraph("RETENTION STRATEGIES", styles['Heading2']))
        notes_style = ParagraphStyle(
            'NotesStyle',
            parent=styles['Normal'],
            backColor=colors.HexColor('#D4EDDA'),
            borderPadding=10,
            spaceAfter=12
        )
        story.append(Paragraph(retention_notes.replace('\n', '<br/>'), notes_style))
        story.append(Spacer(1, 15))
    
    # Development Plan
    if development_plan and development_plan.strip():
        story.append(Paragraph("DEVELOPMENT PLAN", styles['Heading2']))
        dev_style = ParagraphStyle(
            'DevStyle',
            parent=styles['Normal'],
            backColor=colors.HexColor('#D1ECF1'),
            borderPadding=10,
            spaceAfter=12
        )
        story.append(Paragraph(development_plan.replace('\n', '<br/>'), dev_style))
        story.append(Spacer(1, 15))
    
    # Risk Factors
    if risk_factors and risk_factors.strip():
        story.append(Paragraph("KEY RISK FACTORS", styles['Heading2']))
        risk_style = ParagraphStyle(
            'RiskStyle',
            parent=styles['Normal'],
            backColor=colors.HexColor('#F8D7DA'),
            borderPadding=10,
            spaceAfter=12
        )
        story.append(Paragraph(risk_factors.replace('\n', '<br/>'), risk_style))
        story.append(Spacer(1, 15))
    
    # Risk Level Explanation
    story.append(Paragraph("RISK LEVEL INTERPRETATION", styles['Heading2']))
    risk_explanation = """
    <b>Low Risk (0-20%):</b> Stable employee. Continue current engagement strategies.<br/>
    <b>Moderate Risk (20-40%):</b> Monitor closely. Implement retention measures.<br/>
    <b>High Risk (40-60%):</b> High attrition risk. Immediate intervention required.<br/>
    <b>Critical Risk (60-100%):</b> Very high risk. Urgent retention actions needed.
    """
    story.append(Paragraph(risk_explanation, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def create_download_link(pdf_buffer, filename):
    """Create a download link for the PDF file."""
    b64 = base64.b64encode(pdf_buffer.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}" style="\
        background-color: #4CAF50;\
        color: white;\
        padding: 12px 24px;\
        text-align: center;\
        text-decoration: none;\
        display: inline-block;\
        border-radius: 4px;\
        font-weight: bold;\
        border: none;\
        cursor: pointer;">\
        📄 Download HR Report</a>'
    return href

# =============================================================================
# UI HELPER FUNCTIONS
# =============================================================================
def create_risk_gauge(probability):
    """Create a Plotly risk gauge visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': "Attrition Probability", 
            'font': {'size': 20, 'color': 'darkblue'}
        },
        gauge={
            'axis': {
                'range': [None, 100],
                'tickwidth': 1,
                'tickcolor': "darkblue"
            },
            'bar': {'color': "darkred"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': 'lightgreen'},
                {'range': [20, 40], 'color': 'yellow'},
                {'range': [40, 60], 'color': 'orange'},
                {'range': [60, 100], 'color': 'red'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    fig.update_layout(
        height=250, 
        margin=dict(l=10, r=10, t=50, b=10),
        font={'color': "darkblue", 'family': "Arial"}
    )
    return fig

def create_risk_barchart(probability, risk_level):
    """Create a horizontal bar chart showing risk level comparison."""
    risk_levels = ['Low Risk', 'Moderate Risk', 'High Risk', 'Critical Risk']
    risk_ranges = ['0-20%', '20-40%', '40-60%', '60-100%']
    risk_colors = ['lightgreen', 'yellow', 'orange', 'red']
    
    values = [20, 20, 20, 20]
    current_risk_index = risk_levels.index(risk_level)
    
    fig = go.Figure()
    
    for i, (level, range_val, color, value) in enumerate(zip(risk_levels, risk_ranges, risk_colors, values)):
        fig.add_trace(go.Bar(
            y=[f"{level}\n{range_val}"],
            x=[value],
            orientation='h',
            marker=dict(
                color=color,
                line=dict(color='black', width=1 if i != current_risk_index else 3)
            ),
            name=level,
            hovertemplate=f"<b>{level}</b><br>Range: {range_val}<extra></extra>"
        ))
    
    fig.add_trace(go.Scatter(
        y=[f"{risk_levels[current_risk_index]}\n{risk_ranges[current_risk_index]}"],
        x=[10],
        mode='markers+text',
        marker=dict(
            size=20,
            color='black',
            symbol='diamond'
        ),
        text=["📍 CURRENT"],
        textposition="middle right",
        textfont=dict(color='black', size=12, weight='bold'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=dict(
            text="👥 Attrition Risk Level Comparison",
            font=dict(size=16, color='darkblue', weight='bold')
        ),
        xaxis=dict(
            title="Risk Scale",
            range=[0, 25],
            showticklabels=False,
            showgrid=False
        ),
        yaxis=dict(
            title="",
            showgrid=False
        ),
        showlegend=False,
        height=200,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='white'
    )
    
    return fig

def create_feature_importance_chart(input_data, probability):
    """Create a bar chart showing feature contributions to the risk score."""
    features = list(input_data.keys())
    values = list(input_data.values())
    
    # Calculate importance scores based on feature impact
    scores = []
    for feature, value in input_data.items():
        if feature in ['job_satisfaction', 'environment_satisfaction', 'work_life_balance']:
            # Lower values = higher risk
            score = max(0, (3 - value) * 0.8)
        elif feature in ['overtime', 'years_since_last_promotion']:
            # Higher values = higher risk
            score = min(3, value * 0.6)
        elif feature in ['job_involvement', 'monthly_income', 'training_times_last_year']:
            # Higher values = lower risk
            score = max(0, (5 - value) * 0.4)
        else:
            score = value * 0.3
            
        scores.append(min(3, score))
    
    importance_df = pd.DataFrame({
        'Feature': [f.replace('_', ' ').title() for f in features],
        'Importance': scores,
        'Value': values
    })
    
    importance_df = importance_df.sort_values('Importance', ascending=True)
    
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='📊 Feature Contribution to Attrition Risk',
        color='Importance',
        color_continuous_scale=['lightgreen', 'yellow', 'orange', 'red'],
        hover_data={'Value': ':.2f'}
    )
    
    fig.update_layout(
        height=300,
        xaxis_title="Risk Contribution Score",
        yaxis_title="",
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    return fig

def show_tutorial():
    """Show the tutorial expander if it hasn't been hidden by the user."""
    if st.session_state.show_tutorial:
        with st.expander("🎓 HR Analytics Quick Start Guide - Click to Expand", expanded=True):
            st.markdown("""
            ### Welcome to the HR Attrition Analytics Pro! 👥

            **Getting Started:**
            
            1.  **Create a New Employee**: Use the 'New Employee' button
            2.  **Enter Employee Metrics**: Fill in the engagement and satisfaction metrics
            3.  **Analyze**: Click 'Analyze Employee' to get attrition risk assessment
            4.  **Review**: Check the prediction, risk level, and retention strategies
            5.  **Customize**: Edit the retention notes and development plans as needed
            6.  **Export**: Download PDF reports for HR records
            7.  **Manage**: View all employee records and their history in the 'Employee Management' tab

            **💾 Database Backend**: This app uses SQLite database and can handle 10,000+ employees efficiently.

            ---
            """)
            
            if st.button(
                "✅ Got it! Hide this guide", 
                key="hide_tutorial_button",
                use_container_width=True
            ):
                st.session_state.show_tutorial = False
                st.rerun()

def create_new_employee_form(unique_suffix=""):
    """Displays a form to create a new employee record with unique key."""
    
    if not st.session_state.show_new_employee_form:
        return
    
    st.subheader("📝 Create New Employee Record")
    
    form_key = f"new_employee_form_{unique_suffix}"
    
    with st.form(form_key, clear_on_submit=True):
        name = st.text_input(
            "Full Name*", 
            placeholder="Enter employee's full name",
            help="Required field"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input(
                "Age", 
                min_value=18, 
                max_value=70, 
                value=35,
                help="Employee's age in years"
            )
            department = st.selectbox(
                "Department",
                ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations", "IT", "Customer Service"],
                help="Employee's department"
            )
            salary = st.number_input(
                "Monthly Salary ($)",
                min_value=1000,
                max_value=50000,
                value=5000,
                step=500,
                help="Gross monthly salary"
            )
        
        with col2:
            position = st.text_input(
                "Position", 
                placeholder="Job title",
                help="Employee's job position"
            )
            contact = st.text_input(
                "Contact Info", 
                placeholder="Email or phone",
                help="Primary contact information"
            )
            hire_date = st.date_input(
                "Hire Date",
                value=datetime.now(),
                help="Date employee was hired"
            )
        
        notes = st.text_area(
            "HR Notes", 
            placeholder="Performance notes, achievements, concerns...",
            help="Additional HR notes and observations"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            submit_btn = st.form_submit_button(
                "✅ Create Employee", 
                use_container_width=True
            )
        with col2:
            cancel_btn = st.form_submit_button(
                "❌ Cancel", 
                use_container_width=True
            )
        
        if submit_btn:
            if name.strip():
                employee_id = f"EMP{str(uuid.uuid4())[:6].upper()}"
                employee_data = {
                    'employee_id': employee_id, 
                    'name': name.strip(), 
                    'age': age,
                    'department': department, 
                    'position': position,
                    'salary': salary,
                    'contact': contact, 
                    'hire_date': hire_date.strftime("%Y-%m-%d"),
                    'notes': notes, 
                    'created_date': datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                
                # Save to database
                save_employee(employee_data)
                
                st.session_state.current_employee_id = employee_id
                st.session_state.show_new_employee_form = False
                
                st.success(f"✅ Employee '{name}' created successfully! ID: {employee_id}")
                st.rerun()
            else:
                st.error("❌ Employee name is required.")
        
        if cancel_btn:
            st.session_state.show_new_employee_form = False
            st.rerun()

def employee_selection_section():
    """UI section for selecting an existing employee with search and pagination."""
    st.subheader("👥 Employee Selection")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Search functionality
        search_term = st.text_input(
            "🔍 Search employees by name, ID, department, or position...",
            value=st.session_state.search_term,
            key="employee_search"
        )
        
        if search_term != st.session_state.search_term:
            st.session_state.search_term = search_term
            st.session_state.current_page = 0  # Reset to first page when searching
        
        # Get employees with pagination
        offset = st.session_state.current_page * st.session_state.employees_per_page
        employees_df = search_employees(
            search_term=st.session_state.search_term,
            limit=st.session_state.employees_per_page,
            offset=offset
        )
        
        # Create employee options for dropdown
        if not employees_df.empty:
            employee_options = {
                f"{row['name']} - {row['department']} (ID: {row['employee_id']})": row['employee_id'] 
                for _, row in employees_df.iterrows()
            }
            
            # Find current employee for default selection
            current_employee_display = next(
                (k for k, v in employee_options.items() 
                 if v == st.session_state.current_employee_id), 
                None
            )

            selected_display = st.selectbox(
                "Choose Employee",
                options=["Select an employee..."] + list(employee_options.keys()),
                index=(
                    list(employee_options.keys()).index(current_employee_display) + 1 
                    if current_employee_display else 0
                ),
                key="employee_selector"
            )
            
            if selected_display != "Select an employee...":
                st.session_state.current_employee_id = employee_options[selected_display]
            else:
                st.session_state.current_employee_id = None
            
            # Pagination controls
            total_employees = get_employee_count()
            total_pages = max(1, (total_employees + st.session_state.employees_per_page - 1) // st.session_state.employees_per_page)
            
            col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
            with col1:
                if st.session_state.current_page > 0:
                    if st.button("⬅️ Previous", key="prev_page"):
                        st.session_state.current_page -= 1
                        st.rerun()
            with col2:
                st.write(f"Page {st.session_state.current_page + 1} of {total_pages}")
            with col3:
                st.write(f"Total employees: {total_employees}")
            with col4:
                if st.session_state.current_page < total_pages - 1:
                    if st.button("Next ➡️", key="next_page"):
                        st.session_state.current_page += 1
                        st.rerun()
        else:
            st.info("No employees found. Create a new employee to get started.")
            st.session_state.current_employee_id = None
            
    with col2:
        if st.button(
            "➕ New Employee", 
            key="new_employee_button", 
            use_container_width=True
        ):
            st.session_state.show_new_employee_form = True
            st.rerun()
    
    if st.session_state.current_employee_id:
        show_current_employee_info()

def show_current_employee_info():
    """Display a summary card for the currently selected employee."""
    employee = get_employee(st.session_state.current_employee_id)
    if not employee:
        st.session_state.current_employee_id = None
        return

    with st.container(border=True):
        st.markdown(f"### 👤 Current Employee: {employee['name']}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**Employee ID:** {employee['employee_id']}")
            st.write(f"**Department:** {employee['department']}")
            st.write(f"**Position:** {employee['position']}")
        
        with col2:
            st.write(f"**Age:** {employee['age']}")
            st.write(f"**Salary:** ${employee['salary']:,.2f}")
            st.write(f"**Hire Date:** {employee['hire_date']}")
        
        with col3:
            st.write(f"**Contact:** {employee.get('contact', 'N/A')}")
            # Get prediction count from database
            pred_count = len(get_employee_predictions(employee['employee_id']))
            st.metric("Risk Assessments", pred_count)

# =============================================================================
# TAB FUNCTIONS
# =============================================================================
def single_prediction_tab(predictor, threshold):
    """Content for the 'Single Employee Analysis' tab."""
    st.header("🔍 Single Employee Analysis")
    
    employee_selection_section()
    
    create_new_employee_form("single_prediction")
    
    if st.session_state.current_employee_id:
        st.subheader("📊 Employee Engagement Metrics")
        
        with st.form("feature_input_form_single", clear_on_submit=False):
            cols = st.columns(3)
            input_data = {}
            features = list(st.session_state.inputs.keys())
            
            for i, feature in enumerate(features):
                with cols[i % 3]:
                    display_name = feature.replace('_', ' ').title()
                    
                    if feature in ['job_satisfaction', 'environment_satisfaction', 'work_life_balance', 'job_involvement']:
                        # 1-5 scale for satisfaction metrics
                        value = st.slider(
                            display_name, 
                            min_value=1.0, 
                            max_value=5.0, 
                            value=float(st.session_state.inputs.get(feature, 3.0)),
                            step=0.1,
                            key=f"input_{feature}_single",
                            help=f"1 (Very Low) to 5 (Very High)"
                        )
                    elif feature == 'monthly_income':
                        value = st.number_input(
                            display_name,
                            min_value=1000.0,
                            max_value=50000.0,
                            value=float(st.session_state.inputs.get(feature, 5000.0)),
                            step=500.0,
                            key=f"input_{feature}_single",
                            format="%.0f"
                        )
                    elif feature == 'overtime':
                        value = st.selectbox(
                            "Works Overtime Frequently",
                            options=[0.0, 1.0],
                            format_func=lambda x: "No" if x == 0.0 else "Yes",
                            index=int(st.session_state.inputs.get(feature, 0.0)),
                            key=f"input_{feature}_single"
                        )
                    else:
                        value = st.number_input(
                            display_name, 
                            value=float(st.session_state.inputs.get(feature, 0.0)),
                            key=f"input_{feature}_single", 
                            step=0.1, 
                            format="%.1f"
                        )
                    input_data[feature] = value
            
            analyze_button = st.form_submit_button(
                "🎯 Analyze Employee Risk", 
                type="primary", 
                use_container_width=True
            )

        if analyze_button:
            st.session_state.inputs.update(input_data)
            input_df = pd.DataFrame([input_data])
            
            with st.spinner("🔬 Analyzing employee metrics..."):
                preds, probs, risks = predictor.predict(input_df, threshold)
                
                st.session_state.single_pred = {
                    'pred': preds, 
                    'probs': probs, 
                    'risks': risks,
                    'input_data': input_data.copy()
                }
                
                # Save prediction to database
                prediction_record = {
                    'employee_id': st.session_state.current_employee_id,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'prediction': "High Attrition Risk" if preds[0] == 1 else "Low Attrition Risk",
                    'probability': float(probs[0][1] * 100),
                    'risk_level': risks[0][0],
                    'threshold': threshold,
                    'features': {k: float(v) for k, v in input_data.items()}
                }
                save_prediction(prediction_record)

        if st.session_state.single_pred:
            pred_data = st.session_state.single_pred
            preds, probs, risks = pred_data['pred'], pred_data['probs'], pred_data['risks']
            prediction_label = "High Attrition Risk" if preds[0] == 1 else "Low Attrition Risk"
            risk_level, probability = risks[0][0], probs[0][1]
            
            # Display prediction result with PDF download button
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                if preds[0] == 1:
                    st.error(
                        f"🚨 **PREDICTION:** {prediction_label} | "
                        f"**RISK LEVEL:** {risk_level} | "
                        f"**PROBABILITY:** {probability:.1%}"
                    )
                else:
                    st.success(
                        f"✅ **PREDICTION:** {prediction_label} | "
                        f"**RISK LEVEL:** {risk_level} | "
                        f"**PROBABILITY:** {probability:.1%}"
                    )
            
            with col2:
                st.metric("Confidence Score", f"{max(probs[0])*100:.1f}%")
            
            with col3:
                # Generate and display PDF download link
                employee_data = get_employee(st.session_state.current_employee_id)
                pdf_buffer = generate_pdf_report(
                    employee_data,
                    pred_data,
                    pred_data['input_data'],
                    st.session_state.retention_notes_text,
                    st.session_state.development_plan_text,
                    st.session_state.risk_factors_text
                )
                
                filename = f"HR_Analysis_Report_{employee_data['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                st.markdown(create_download_link(pdf_buffer, filename), unsafe_allow_html=True)

            # Risk visualization section
            st.subheader("📈 Risk Assessment Visualization")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                st.plotly_chart(
                    create_risk_gauge(probability), 
                    use_container_width=True
                )
            
            with viz_col2:
                st.plotly_chart(
                    create_risk_barchart(probability, risk_level),
                    use_container_width=True
                )
            
            st.plotly_chart(
                create_feature_importance_chart(pred_data['input_data'], probability),
                use_container_width=True
            )
            
            # HR strategies and notes
            st.subheader("💼 Retention Strategies & Development Plans")
            
            updated_retention_notes = st.text_area(
                "Retention Strategies & Notes", 
                value=st.session_state.retention_notes_text, 
                key="retention_notes_area", 
                height=100
            )
            
            col1, col2 = st.columns(2)
            with col1:
                updated_development_plan = st.text_area(
                    "Development Plan", 
                    value=st.session_state.development_plan_text, 
                    key="development_plan_area", 
                    height=120
                )
            with col2:
                updated_risk_factors = st.text_area(
                    "Key Risk Factors", 
                    value=st.session_state.risk_factors_text, 
                    key="risk_factors_area", 
                    height=120
                )
            
            # Save settings when changed
            if (updated_retention_notes != st.session_state.retention_notes_text or
                updated_development_plan != st.session_state.development_plan_text or
                updated_risk_factors != st.session_state.risk_factors_text):
                
                st.session_state.retention_notes_text = updated_retention_notes
                st.session_state.development_plan_text = updated_development_plan
                st.session_state.risk_factors_text = updated_risk_factors
                
                # Save to database
                save_hr_settings({
                    'retention_notes_text': updated_retention_notes,
                    'development_plan_text': updated_development_plan,
                    'risk_factors_text': updated_risk_factors
                })
            
            # Recommended actions based on risk level
            st.subheader("🎯 Recommended HR Actions")
            if risk_level == "Critical Risk":
                st.error("""
                **Immediate Actions Required:**
                - Schedule urgent one-on-one meeting with manager
                - Conduct stay interview
                - Review compensation and benefits package
                - Develop personalized retention plan
                - Consider role modification or promotion
                """)
            elif risk_level == "High Risk":
                st.warning("""
                **Priority Actions:**
                - Schedule manager check-in within 1 week
                - Assess career development opportunities
                - Review workload and work-life balance
                - Consider skill development programs
                - Monitor engagement closely
                """)
            elif risk_level == "Moderate Risk":
                st.info("""
                **Proactive Measures:**
                - Regular performance conversations
                - Career path discussions
                - Training and development opportunities
                - Team building activities
                - Quarterly check-ins
                """)
            else:
                st.success("""
                **Maintenance Actions:**
                - Continue regular engagement activities
                - Annual career development reviews
                - Maintain competitive compensation
                - Foster positive work environment
                - Recognize achievements
                """)
                
            # Refresh PDF button
            st.info("💡 **Note**: If you update the retention strategies or development plans, click the button below to refresh the PDF report.")
            if st.button("🔄 Refresh PDF Report with Updated Plans", key="refresh_pdf_button"):
                # Regenerate PDF with updated notes
                pdf_buffer = generate_pdf_report(
                    employee_data,
                    pred_data,
                    pred_data['input_data'],
                    st.session_state.retention_notes_text,
                    st.session_state.development_plan_text,
                    st.session_state.risk_factors_text
                )
                
                filename = f"HR_Analysis_Report_{employee_data['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                st.markdown(create_download_link(pdf_buffer, filename), unsafe_allow_html=True)
                st.success("PDF report updated with latest strategies!")
    else:
        st.info("👆 Please select or create an employee to begin risk analysis.")

def batch_prediction_tab(predictor, threshold):
    """Content for the 'Batch Employee Analysis' tab."""
    st.header("📁 Batch Employee Analysis")
    
    st.info(
        "Upload a CSV or Excel file with employee data. "
        "The file must contain columns matching the required HR metrics."
    )
    
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=['csv', 'xlsx'], 
        key="batch_file_uploader"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} employee records.")
            
            with st.expander("📊 Data Preview", expanded=True):
                st.dataframe(df.head())
            
            if st.button(
                "🔍 Analyze All Employees", 
                type="primary", 
                key="batch_analyze_button"
            ):
                with st.spinner(f"Analyzing {len(df)} employee records..."):
                    preds, probs, risks = predictor.predict(df, threshold)
                    
                    results_df = df.copy()
                    results_df['Prediction'] = [
                        'High Attrition Risk' if p == 1 else 'Low Attrition Risk' for p in preds
                    ]
                    results_df['Risk_Level'] = [r[0] for r in risks]
                    results_df['Attrition_Probability'] = [p[1] * 100 for p in probs]
                    
                    st.session_state.batch_pred = {'df': results_df}

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

    if st.session_state.batch_pred:
        results_df = st.session_state.batch_pred['df']
        
        st.subheader("📈 Analysis Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Employees", len(results_df))
        with col2:
            low_risk_count = (results_df['Prediction'] == 'Low Attrition Risk').sum()
            st.metric("Low Risk Employees", low_risk_count)
        with col3:
            st.metric("High Risk Employees", len(results_df) - low_risk_count)
        with col4:
            critical_risk_count = (results_df['Risk_Level'] == 'Critical Risk').sum()
            st.metric("Critical Risk Cases", critical_risk_count)
        
        st.subheader("📋 Detailed Results")
        st.dataframe(results_df)
        
        st.subheader("📊 Risk Distribution")
        risk_counts = results_df['Risk_Level'].value_counts()
        
        fig = Figure(figsize=(12, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        
        ax1.pie(
            risk_counts.values, 
            labels=risk_counts.index, 
            autopct='%1.1f%%', 
            startangle=90,
            colors=['lightgreen', 'yellow', 'orange', 'red']
        )
        ax1.set_title('Risk Level Distribution', fontweight='bold')
        
        risk_counts.sort_index().plot(
            kind='bar', 
            ax=ax2, 
            color=['lightgreen', 'yellow', 'orange', 'red']
        )
        ax2.set_title('Risk Level Counts', fontweight='bold')
        ax2.set_ylabel('Number of Employees')
        ax2.tick_params(axis='x', rotation=45)
        fig.tight_layout()
        
        st.pyplot(fig)

def employee_management_tab():
    """Content for the 'Employee Management & History' tab."""
    st.header("📋 Employee Management & History")
    
    if st.button(
        "➕ Create New Employee", 
        key="management_new_employee_button", 
        use_container_width=True
    ):
        st.session_state.show_new_employee_form = True
        st.rerun()
    
    create_new_employee_form("employee_management")
    st.divider()

    # Employee search and pagination
    st.subheader("🔍 Employee Search")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_term = st.text_input(
            "Search by name, ID, department, or position...",
            key="management_search"
        )
    with col2:
        employees_per_page = st.selectbox(
            "Employees per page",
            [10, 20, 50, 100],
            index=1,
            key="management_employees_per_page"
        )
    
    # Get employees with pagination
    offset = st.session_state.current_page * employees_per_page
    employees_df = search_employees(
        search_term=search_term,
        limit=employees_per_page,
        offset=offset
    )
    
    total_employees = get_employee_count()
    
    if employees_df.empty:
        st.info("📝 No employee records found. Create an employee to get started.")
        return
    
    st.subheader(f"👥 Employee Records ({total_employees} total)")
    
    # Display employees
    for _, employee_row in employees_df.iterrows():
        with st.expander(f"**{employee_row['name']}** - {employee_row['department']} (ID: {employee_row['employee_id']})"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Age", employee_row['age'])
                st.metric("Position", employee_row['position'])
            with col2:
                st.metric("Salary", f"${employee_row['salary']:,.2f}")
                st.metric("Hire Date", employee_row['hire_date'])
            with col3:
                pred_count = len(get_employee_predictions(employee_row['employee_id']))
                st.metric("Risk Assessments", pred_count)
                st.metric("Department", employee_row['department'])
            
            if employee_row['notes']:
                st.write("**HR Notes:**")
                st.info(employee_row['notes'])
                
            # Get prediction history
            predictions = get_employee_predictions(employee_row['employee_id'], limit=5)
            if not predictions.empty:
                st.write("**Risk Assessment History (Latest 5):**")
                st.dataframe(
                    predictions[['timestamp', 'prediction', 'risk_level', 'probability']],
                    use_container_width=True
                )
            else:
                st.info("No risk assessments recorded for this employee yet.")
    
    # Pagination controls
    total_pages = max(1, (total_employees + employees_per_page - 1) // employees_per_page)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.session_state.current_page > 0:
            if st.button("⬅️ Previous", key="management_prev"):
                st.session_state.current_page -= 1
                st.rerun()
    with col2:
        st.write(f"Page {st.session_state.current_page + 1} of {total_pages}")
    with col3:
        if st.session_state.current_page < total_pages - 1:
            if st.button("Next ➡️", key="management_next"):
                st.session_state.current_page += 1
                st.rerun()

def about_tab():
    """Content for the 'About' tab."""
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ## HR Attrition Analytics Pro 👥

    This application is a comprehensive tool designed to assist HR professionals 
    in analyzing employee engagement metrics and predicting attrition risk based on key factors.

    ### 🎯 Purpose
    - Provide data-driven attrition risk assessment for employees
    - Support HR decision-making with actionable insights  
    - Maintain a robust system for employee records and risk history
    - Generate comprehensive PDF reports for HR documentation

    ### 🔑 Key Features
    - **Single Employee Analysis**: Individual risk assessment with PDF export
    - **Batch Processing**: Analyze multiple employees from files
    - **Employee Management**: Comprehensive record keeping with SQLite database
    - **Visual Analytics**: Interactive risk gauges and charts
    - **Report Generation**: Professional PDF reports with customizable strategies
    - **Scalable Database**: Supports 10,000+ employee records efficiently
    - **Secure Admin Access**: Password-protected data export features

    ### 📊 Risk Factors Analyzed
    - Job Satisfaction & Engagement
    - Work Environment & Culture
    - Work-Life Balance
    - Career Development & Growth
    - Compensation & Benefits
    - Tenure & Promotion History

    ---
    
    *Built with Streamlit for HR analytics and workforce planning.*
    """)

# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    """Main application function to run the Streamlit app."""
    
    st.title("👥 HR Attrition Analytics Pro")
    st.markdown(
        "Analyze employee engagement metrics to predict attrition risk and support retention strategies."
    )
    
    show_tutorial()
    
    # Authentication system
    is_logged_in = login_section()
    
    # Data export system (only for logged-in users)
    download_all_data_section()
    
    with st.sidebar:
        st.header("⚙️ HR Configuration")
        
        threshold = st.slider(
            "Risk Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5, 
            step=0.05,
            help="Adjust the sensitivity for high-risk classification. Higher values are more strict.",
            key="risk_threshold"
        )
        
        st.header("📊 Risk Categories")
        st.info("""
        **Risk Levels:**
        - 🟢 **Low Risk**: 0-20%
        - 🟡 **Moderate Risk**: 20-40%  
        - 🟠 **High Risk**: 40-60%
        - 🔴 **Critical Risk**: 60-100%
        """)
        
        st.header("🚀 Quick Actions")
        
        st.warning("The action below will reset all session data (but keeps database records).")
        if st.button("🔄 Reset Session Data", key="reset_session_button"):
            keys_to_reset = [
                'current_employee_id', 'single_pred', 'batch_pred', 
                'show_new_employee_form', 'current_page', 'search_term'
            ]
            for key in keys_to_reset:
                if key in st.session_state:
                    del st.session_state[key]
            initialize_session_state()
            st.success("Session data reset!")
            st.rerun()

        if st.button("📖 Show Tutorial", key="sidebar_show_tutorial_button"):
            st.session_state.show_tutorial = True
            st.rerun()
        
        st.header("📈 HR Analytics Status")
        employee_count = get_employee_count()
        st.metric("Employees in Database", employee_count)
        
        st.header("💾 Database Status")
        if os.path.exists(DB_FILE):
            file_size = os.path.getsize(DB_FILE)
            st.success(f"✅ SQLite Database")
            st.info(f"File size: {file_size:,} bytes")
            
            # Show database info
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            
            c.execute('SELECT COUNT(*) FROM predictions')
            pred_count = c.fetchone()[0]
            
            c.execute('SELECT COUNT(*) FROM hr_settings')
            settings_count = c.fetchone()[0]
            
            conn.close()
            
            st.write(f"Risk Assessments: {pred_count}")
            st.write(f"HR Settings: {settings_count}")
        else:
            st.warning("❌ Database file not found")
        
        if st.session_state.current_employee_id:
            current_employee = get_employee(st.session_state.current_employee_id)
            if current_employee:
                st.success(f"Selected: {current_employee['name']}")
        else:
            st.warning("No employee selected")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Single Analysis", 
        "📁 Batch Analysis", 
        "📋 Employee Management", 
        "ℹ️ About"
    ])
    
    with tab1:
        single_prediction_tab(AttritionPredictor(), threshold)
    with tab2:
        batch_prediction_tab(AttritionPredictor(), threshold)
    with tab3:
        employee_management_tab()
    with tab4:
        about_tab()

if __name__ == "__main__":
    main()