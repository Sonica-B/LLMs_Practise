from LLMs_Practise.LLMs_Practise.TriageSystem import MedicalTriageSystem
import os
import json
from datetime import datetime


class TriageSystemTester:
    def __init__(self):
        self.triage = MedicalTriageSystem()
        self.results = []

    def read_case_file(self, file_path):
        """Read a case file and return its content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            return None

    def test_single_case(self, case_id, case_text):
        """Test a single case and return results"""
        try:
            # Parse and process the case
            case_data = self.triage.parse_medical_case(case_text)
            esi_result = self.triage.determine_esi_level(case_data)
            recommendations = self.triage.generate_recommendations(esi_result, case_data)

            # Compile test results
            result = {
                'case_id': case_id,
                'chief_complaint': case_data.get('chief_complaint', ''),
                'esi_level': esi_result['esi_level'],
                'confidence': round(esi_result['confidence'], 2),
                'needs_handoff': esi_result['needs_handoff'],
                'risk_factors': esi_result.get('risk_factors', []),
                'wait_time': recommendations['estimated_wait_time'],
                'original_acuity': case_data.get('visit_info', {}).get('Acuity', 'Not specified'),
                'arrival_method': case_data.get('visit_info', {}).get('Arrival Transport', 'Not specified'),
                'disposition': case_data.get('visit_info', {}).get('Disposition', 'Not specified')
            }

            return result

        except Exception as e:
            print(f"Error processing case {case_id}: {str(e)}")
            return None

    def run_batch_test(self, data_dir):
        """Run tests on all case files in directory"""
        for filename in os.listdir(data_dir):
            if filename.startswith('3000') and filename.endswith('.txt'):
                case_id = filename.replace('.txt', '')
                file_path = os.path.join(data_dir, filename)

                print(f"\nProcessing case {case_id}...")
                case_text = self.read_case_file(file_path)

                if case_text:
                    result = self.test_single_case(case_id, case_text)
                    if result:
                        self.results.append(result)
                        self._print_case_summary(result)

    def _print_case_summary(self, result):
        """Print a summary of case results"""
        print("\nCase Summary:")
        print(f"Case ID: {result['case_id']}")
        print(f"Chief Complaint: {result['chief_complaint'][:100]}...")
        print(f"ESI Level: {result['esi_level']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Handoff Needed: {result['needs_handoff']}")
        print(f"Estimated Wait Time: {result['wait_time']} hours")
        print(f"Original Acuity: {result['original_acuity']}")
        print("\nRisk Factors:")
        for factor in result['risk_factors']:
            print(f"- {factor}")

    def generate_report(self, output_path="triage_test_results.json"):
        """Generate and save test results report"""
        report = {
            'test_date': datetime.now().isoformat(),
            'total_cases': len(self.results),
            'summary': {
                'average_confidence': sum(r['confidence'] for r in self.results) / len(self.results),
                'handoff_rate': sum(1 for r in self.results if r['needs_handoff']) / len(self.results),
                'esi_distribution': {
                    level: sum(1 for r in self.results if r['esi_level'] == level)
                    for level in range(1, 6)
                }
            },
            'cases': self.results
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nTest Report Summary:")
        print(f"Total Cases Processed: {report['total_cases']}")
        print(f"Average Confidence: {report['summary']['average_confidence']:.2f}")
        print(f"Handoff Rate: {report['summary']['handoff_rate'] * 100:.1f}%")
        print("\nESI Level Distribution:")
        for level, count in report['summary']['esi_distribution'].items():
            print(f"ESI {level}: {count} cases ({count / report['total_cases'] * 100:.1f}%)")


# Run tests
if __name__ == "__main__":
    tester = TriageSystemTester()
    data_dir = "D:\\WPI Assignments\\RA-Shraga\\patient_record_example\\patient_record_example"

    print("Starting triage system tests...")
    tester.run_batch_test(data_dir)
    tester.generate_report()
    print("\nTesting completed!")