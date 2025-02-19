import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import numpy as np
from datetime import datetime
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')


class MedicalRecordsEDA:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.records = []
        self.stats = {}
        self.df = None

        # Set up plotting style
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['figure.dpi'] = 100

    def load_records(self):
        """Load all medical records from the directory"""
        for filename in os.listdir(self.data_dir):
            if filename.startswith('3000') and filename.endswith('.txt'):
                file_path = os.path.join(self.data_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.records.append({
                        'record_id': filename.replace('.txt', ''),
                        'content': f.read()
                    })
        print(f"Loaded {len(self.records)} medical records.")

    def parse_section(self, text, section_name):
        """Extract content of a specific section"""
        pattern = f"==================================== {section_name} ====================================\n(.*?)(?====================================|$)"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def extract_visit_info(self, content):
        """Extract ER visit information"""
        visit_info = {}
        visit_section = self.parse_section(content, "ER Visit Info")

        for line in visit_section.split('\n'):
            if ':' in line:
                key, value = [x.strip() for x in line.split(':', 1)]
                visit_info[key] = value
        return visit_info

    def count_qa_pairs(self, content):
        """Count number of Q&A pairs"""
        qa_section = self.parse_section(content, "Question Answer Pair")
        return len(re.findall(r'Question \d+:', qa_section))

    def analyze_records(self):
        """Perform comprehensive analysis of medical records"""
        # Initialize data structures for record details
        record_details = []

        # Process each record
        for record in self.records:
            content = record['content']

            # Extract visit info
            visit_info = self.extract_visit_info(content)

            # Get chief complaint
            chief_complaint = self.parse_section(content, "Patient Chief Complaint")

            # Count Q&A pairs
            qa_count = self.count_qa_pairs(content)

            # Count risk terms
            risk_terms = ['pain', 'severe', 'acute', 'critical', 'emergency',
                          'unconscious', 'bleeding', 'trauma', 'chest', 'breathing']
            risk_count = sum(1 for term in risk_terms if term in content.lower())

            # Store record details
            record_details.append({
                'record_id': record['record_id'],
                'diagnosis': visit_info.get('Diagnosis', 'Unknown'),
                'acuity': float(visit_info.get('Acuity', 0)),
                'transport': visit_info.get('Arrival Transport', 'Unknown'),
                'disposition': visit_info.get('Disposition', 'Unknown'),
                'complaint_length': len(chief_complaint.split()),
                'qa_count': qa_count,
                'risk_terms': risk_count
            })

        # Convert to DataFrame
        self.df = pd.DataFrame(record_details)

        # Add severity score calculation
        self.calculate_severity_scores()

        # Generate basic statistics
        self.generate_basic_stats()

        # Perform advanced analysis
        self.perform_advanced_analysis()

    def calculate_severity_scores(self):
        """Calculate severity scores for all patients"""
        self.df['severity_score'] = 0

        # Add points based on risk terms
        self.df.loc[self.df['risk_terms'] >= 4, 'severity_score'] += 2
        self.df.loc[self.df['risk_terms'] >= 5, 'severity_score'] += 1

        # Add points based on complaint length
        self.df.loc[self.df['complaint_length'] > self.df['complaint_length'].median(),
        'severity_score'] += 1

        # Add points based on acuity
        self.df.loc[self.df['acuity'] == 2.0, 'severity_score'] += 2

        # Create severity categories
        self.df['severity_category'] = pd.cut(self.df['severity_score'],
                                              bins=[-np.inf, 1, 3, np.inf],
                                              labels=['Low', 'Medium', 'High'])

    def perform_advanced_analysis(self):
        """Perform advanced statistical analysis"""
        # Transport and admission analysis
        transport_admission = pd.crosstab(self.df['transport'],
                                          self.df['disposition'] == 'ADMITTED')
        odds_ratio, p_value = fisher_exact(transport_admission)

        # Risk terms analysis
        X = self.df[['risk_terms', 'complaint_length', 'acuity']]
        y = (self.df['disposition'] == 'ADMITTED')

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)

        # Store advanced analysis results
        self.advanced_stats = {
            'transport_admission': {
                'contingency': transport_admission,
                'odds_ratio': odds_ratio,
                'p_value': p_value
            },
            'feature_importance': pd.DataFrame({
                'feature': ['Risk Terms', 'Complaint Length', 'Acuity'],
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
        }

    def generate_basic_stats(self):
        """Generate basic statistics"""
        self.stats = {
            'total_records': len(self.records),
            'diagnoses': self.df['diagnosis'].value_counts(),
            'acuity_levels': self.df['acuity'].value_counts(),
            'transport_methods': self.df['transport'].value_counts(),
            'dispositions': self.df['disposition'].value_counts(),
            'numeric_stats': {
                'complaint_length': self.df['complaint_length'].describe(),
                'qa_count': self.df['qa_count'].describe(),
                'risk_terms': self.df['risk_terms'].describe()
            }
        }

    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        os.makedirs('eda_output', exist_ok=True)

        # 1. Basic Visualizations
        self._create_basic_visualizations()

        # 2. Advanced Visualizations
        self._create_advanced_visualizations()

    def _create_basic_visualizations(self):
        # Acuity Distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x='acuity')
        plt.title('Distribution of Acuity Levels')
        plt.tight_layout()
        plt.savefig('eda_output/acuity_distribution.png')
        plt.close()

        # Transport Methods
        plt.figure(figsize=(10, 6))
        self.df['transport'].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title('Patient Transport Methods')
        plt.tight_layout()
        plt.savefig('eda_output/transport_methods.png')
        plt.close()

        # Complaint Lengths
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x='complaint_length', bins=20)
        plt.title('Distribution of Chief Complaint Lengths')
        plt.xlabel('Word Count')
        plt.tight_layout()
        plt.savefig('eda_output/complaint_lengths.png')
        plt.close()

    def _create_advanced_visualizations(self):
        # Severity by Transport Method
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df, x='transport', y='severity_score')
        plt.title('Severity Scores by Transport Method')
        plt.tight_layout()
        plt.savefig('eda_output/severity_by_transport.png')
        plt.close()

        # Risk Terms vs Disposition
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df, x='disposition', y='risk_terms')
        plt.title('Risk Terms by Disposition')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('eda_output/risk_terms_by_disposition.png')
        plt.close()

        # Feature Importance
        plt.figure(figsize=(10, 6))
        importance_df = self.advanced_stats['feature_importance']
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title('Feature Importance for Admission Prediction')
        plt.tight_layout()
        plt.savefig('eda_output/feature_importance.png')
        plt.close()

    def generate_report(self):
        """Generate comprehensive analysis report"""
        report_text = f"""
Medical Records EDA Report
=========================
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Basic Statistics
---------------
Total Records: {self.stats['total_records']}

Chief Complaint Statistics:
{self.stats['numeric_stats']['complaint_length'].to_string()}

Q&A Interaction Statistics:
{self.stats['numeric_stats']['qa_count'].to_string()}

Risk Terms Statistics:
{self.stats['numeric_stats']['risk_terms'].to_string()}

Transport Method Distribution:
----------------------------
{self.stats['transport_methods'].to_string()}

Disposition Distribution:
------------------------
{self.stats['dispositions'].to_string()}

Acuity Level Distribution:
-------------------------
{self.stats['acuity_levels'].to_string()}

Advanced Analysis
----------------
Transport-Admission Association:
Odds Ratio: {self.advanced_stats['transport_admission']['odds_ratio']:.2f}
P-value: {self.advanced_stats['transport_admission']['p_value']:.4f}

Feature Importance for Admission:
{self.advanced_stats['feature_importance'].to_string()}

Severity Analysis:
{self.df.groupby('severity_category').size().to_string()}
"""

        # Save report
        os.makedirs('eda_output', exist_ok=True)
        with open('eda_output/eda_report.txt', 'w') as f:
            f.write(report_text)

        return self.stats


def main():
    # Initialize analyzer
    eda = MedicalRecordsEDA("D:\\WPI Assignments\\RA-Shraga\\patient_record_example\\patient_record_example")

    # Load and analyze records
    print("Loading records...")
    eda.load_records()

    print("Analyzing records...")
    eda.analyze_records()

    print("Generating visualizations...")
    eda.generate_visualizations()

    print("Generating report...")
    eda.generate_report()

    print("\nAnalysis complete! Check the 'eda_output' directory for results.")


if __name__ == "__main__":
    main()