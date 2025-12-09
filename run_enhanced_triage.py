import os
import json
from datetime import datetime
from enhanced_triage_system import EnhancedMedicalTriageSystem
from triage_evaluation import TriageEvaluationSystem


def read_case_file(file_path):
    """Read a case file and return its content"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return None


def run_enhanced_triage_evaluation(data_dir, output_path=None, max_cases=None):
    """Run enhanced evaluation on triage system using test cases"""
    print("Starting enhanced triage system evaluation...")

    # Initialize the evaluation system
    evaluator = TriageEvaluationSystem()

    # Process test cases
    case_count = 0
    for filename in os.listdir(data_dir):
        if filename.startswith('3000') and filename.endswith('.txt'):
            if max_cases and case_count >= max_cases:
                break

            case_id = filename.replace('.txt', '')
            file_path = os.path.join(data_dir, filename)

            print(f"\nProcessing case {case_id}...")

            # Read case file
            case_text = read_case_file(file_path)
            if not case_text:
                continue

            # Evaluate case
            result = evaluator.evaluate_case(case_id, case_text)
            if result:
                case_count += 1
                evaluator.print_evaluation_summary(result)

    # Generate and print overall metrics
    metrics = evaluator.generate_metrics()
    print(f"\nProcessed {case_count} cases successfully.")
    evaluator.print_evaluation_summary()

    # Save detailed report
    output_file = output_path or f"triage_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    evaluator.generate_detailed_report(output_file)

    return evaluator


def test_single_case(case_text, case_id="test_case"):
    """Test a single case input with the enhanced triage system"""
    print("Evaluating single case...")

    # Initialize system
    triage_system = EnhancedMedicalTriageSystem()

    # Parse and process the case
    case_data = triage_system.parse_medical_case(case_text)
    esi_result = triage_system.determine_esi_level(case_data)
    recommendations = triage_system.generate_recommendations(esi_result, case_data)

    # Print results
    print("\n===== Enhanced Triage Analysis =====")
    print(f"Chief Complaint: {case_data.get('chief_complaint', '')[:100]}...")
    print(f"\nESI Level: {esi_result['esi_level']} (Confidence: {esi_result['confidence']:.2f})")

    # Print handoff decision
    print(f"\nHandoff Required: {esi_result['needs_handoff']}")
    if esi_result['needs_handoff']:
        print(f"Primary Reason: {esi_result['handoff_analysis'].get('primary_reason', 'Not specified')}")
        print("\nHandoff Reasons:")
        for reason in esi_result['handoff_analysis'].get('reasons', []):
            if isinstance(reason, dict):
                print(f"- {reason.get('category', '')}: {reason.get('explanation', '')}")
            else:
                print(f"- {reason}")
    else:
        print("This case can be handled autonomously by the AI system")

    # Print diagnoses
    print("\nPossible Diagnoses:")
    for diagnosis in esi_result.get('possible_diagnoses', [])[:3]:
        print(f"- {diagnosis.get('diagnosis', '')}: {diagnosis.get('likelihood', 'medium')} likelihood")

    # Print risk factors
    print("\nIdentified Risk Factors:")
    for risk in esi_result.get('risk_factors', []):
        print(f"- {risk}")

    # Print recommendations
    print("\nStructured Recommendations:")
    structured_recs = recommendations.get('structured_recommendations', {})

    # Print lab recommendations
    if 'labs' in structured_recs and structured_recs['labs']:
        print("\nRecommended Labs:")
        for lab in structured_recs['labs']:
            if isinstance(lab, dict):
                print(f"- {lab.get('name', '')}: {lab.get('rationale', '')}")
            else:
                print(f"- {lab}")

    # Print imaging recommendations
    if 'imaging' in structured_recs and structured_recs['imaging']:
        print("\nRecommended Imaging:")
        for img in structured_recs['imaging']:
            if isinstance(img, dict):
                print(f"- {img.get('name', '')}: {img.get('rationale', '')}")
            else:
                print(f"- {img}")

    # Print wait time
    print(f"\nEstimated Wait Time: {recommendations['estimated_wait_time']} hours")

    return {
        'esi_result': esi_result,
        'recommendations': recommendations
    }


# Example usage
if __name__ == "__main__":
    # Path to your test data
    data_dir = "D:\\WPI Assignments\\RA-Shraga\\patient_record_example\\patient_record_example"

    # Choose what to run
    mode = "batch"  # Options: "batch", "single"

    if mode == "batch":
        # Run batch evaluation
        evaluator = run_enhanced_triage_evaluation(data_dir, max_cases=10)
    else:
        # Test with a single case input
        sample_case = """
===== Patient Chief Complaint =====
I have severe chest pain radiating to my left arm and jaw. It started about an hour ago.

===== Summary of Current ER Visit =====
Patient is a 58-year-old male presenting with chest pain radiating to left arm and jaw. Pain started 1 hour ago while working in yard. Pain is described as pressure-like and rated 8/10. Patient has history of hypertension and type 2 diabetes. Vital signs: BP 165/95, HR 102, RR 22, O2 sat 96%, temp 37.1C. Patient took 1 aspirin before arrival.

===== Patient History Summary =====
HTN diagnosed 10 years ago, on lisinopril 10mg daily
Type 2 diabetes diagnosed 7 years ago, on metformin 1000mg BID
Cholesterol elevation, on atorvastatin 20mg daily
Father died of MI at age 62
No prior heart attacks or procedures
Smoker, 1 pack per day for 35 years

===== ER Visit Info =====
Arrival Transport: AMBULANCE 
Acuity: 2
Disposition: ADMITTED
        """

        result = test_single_case(sample_case)