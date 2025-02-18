import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import numpy as np
from datetime import datetime


class MedicalRecordsEDA:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.records = []
        self.stats = {}

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
        pattern = f"==== {section_name} ====\n(.*?)(?=====|$)"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def extract_visit_info(self, text):
        """Extract ER visit information"""
        visit_info = {}
        visit_section = self.parse_section(text, "ER Visit Info")

        for line in visit_info.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                visit_info[key.strip()] = value.strip()
        return visit_info

    def analyze_records(self):
        """Perform comprehensive analysis of medical records"""
        # Initialize data structures
        self.stats = {
            'diagnoses': Counter(),
            'acuity_levels': Counter(),
            'transport_methods': Counter(),
            'dispositions': Counter(),
            'qa_counts': [],
            'complaint_lengths': [],
            'high_risk_terms': Counter()
        }

        # High-risk terms to track
        high_risk_terms = ['pain', 'severe', 'acute', 'critical', 'emergency',
                           'unconscious', 'bleeding', 'trauma', 'chest', 'breathing']

        # Process each record
        for record in self.records:
            content = record['content']

            # Extract visit info
            visit_info = {}
            visit_section = self.parse_section(content, "ER Visit Info")
            for line in visit_section.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    visit_info[key.strip()] = value.strip()

            # Update stats
            self.stats['diagnoses'][visit_info.get('Diagnosis', 'Unknown')] += 1
            self.stats['acuity_levels'][visit_info.get('Acuity', 'Unknown')] += 1
            self.stats['transport_methods'][visit_info.get('Arrival Transport', 'Unknown')] += 1
            self.stats['dispositions'][visit_info.get('Disposition', 'Unknown')] += 1

            # Analyze chief complaint
            chief_complaint = self.parse_section(content, "Patient Chief Complaint")
            self.stats['complaint_lengths'].append(len(chief_complaint.split()))

            # Count high-risk terms
            for term in high_risk_terms:
                if term in content.lower():
                    self.stats['high_risk_terms'][term] += 1

            # Count Q&A pairs
            qa_section = self.parse_section(content, "Question Answer Pair")
            qa_count = len(re.findall(r'Question \d+:', qa_section))
            self.stats['qa_counts'].append(qa_count)

    def generate_visualizations(self):
        """Generate visualizations for the analyzed data"""
        # Set style
        plt.style.use('seaborn')

        # 1. Acuity Level Distribution
        plt.figure(figsize=(10, 6))
        acuity_df = pd.DataFrame.from_dict(self.stats['acuity_levels'],
                                           orient='index', columns=['count'])
        acuity_df.plot(kind='bar')
        plt.title('Distribution of Acuity Levels')
        plt.xlabel('Acuity Level')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('acuity_distribution.png')

        # 2. Transport Methods
        plt.figure(figsize=(10, 6))
        transport_df = pd.DataFrame.from_dict(self.stats['transport_methods'],
                                              orient='index', columns=['count'])
        transport_df.plot(kind='pie', y='count', autopct='%1.1f%%')
        plt.title('Patient Transport Methods')
        plt.tight_layout()
        plt.savefig('transport_methods.png')

        # 3. High Risk Terms Frequency
        plt.figure(figsize=(12, 6))
        terms_df = pd.DataFrame.from_dict(self.stats['high_risk_terms'],
                                          orient='index', columns=['count'])
        terms_df.sort_values('count', ascending=True).plot(kind='barh')
        plt.title('Frequency of High-Risk Terms')
        plt.xlabel('Count')
        plt.tight_layout()
        plt.savefig('high_risk_terms.png')

        # 4. Chief Complaint Length Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.stats['complaint_lengths'], bins=20)
        plt.title('Distribution of Chief Complaint Lengths')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('complaint_lengths.png')

        # Clear plots
        plt.close('all')

    def generate_report(self):
        """Generate a summary report of the analysis"""
        report = {
            'total_records': len(self.records),
            'average_complaint_length': np.mean(self.stats['complaint_lengths']),
            'average_qa_count': np.mean(self.stats['qa_counts']),
            'most_common_diagnosis': self.stats['diagnoses'].most_common(1)[0],
            'transport_distribution': dict(self.stats['transport_methods']),
            'acuity_distribution': dict(self.stats['acuity_levels']),
            'disposition_distribution': dict(self.stats['dispositions']),
            'high_risk_terms': dict(self.stats['high_risk_terms'])
        }

        # Format report as text
        report_text = f"""
Medical Records EDA Report
=========================
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Summary Statistics:
------------------
Total Records: {report['total_records']}
Average Chief Complaint Length: {report['average_complaint_length']:.1f} words
Average Q&A Count: {report['average_qa_count']:.1f}

Most Common Diagnosis:
---------------------
{report['most_common_diagnosis'][0]}: {report['most_common_diagnosis'][1]} cases

Transport Method Distribution:
----------------------------
"""
        for method, count in report['transport_distribution'].items():
            report_text += f"{method}: {count} ({count / report['total_records'] * 100:.1f}%)\n"

        report_text += "\nAcuity Level Distribution:\n---------------------------\n"
        for level, count in report['acuity_distribution'].items():
            report_text += f"Level {level}: {count} ({count / report['total_records'] * 100:.1f}%)\n"

        # Save report
        with open('eda_report.txt', 'w') as f:
            f.write(report_text)

        return report


def main():
    # Initialize analyzer
    eda = MedicalRecordsEDA("/patient_record_example/patient_record_example")

    # Load and analyze records
    print("Loading records...")
    eda.load_records()

    print("Analyzing records...")
    eda.analyze_records()

    print("Generating visualizations...")
    eda.generate_visualizations()

    print("Generating report...")
    eda.generate_report()

    print("Analysis complete! Check the output files for results.")


if __name__ == "__main__":
    main()