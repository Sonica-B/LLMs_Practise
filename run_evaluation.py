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


def run_enhanced_triage_evaluation(data_dir, output_path=None):
    """Run enhanced evaluation on triage system using test cases"""
    print("Starting enhanced triage system evaluation...")

    # Initialize the evaluation system
    evaluator = TriageEvaluationSystem()

    # Process test cases
    case_count = 0
    for filename in os.listdir(data_dir):
        if filename.startswith('3000') and filename.endswith('.txt'):
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


def run_comparative_evaluation(data_dir, output_path=None):
    """Run comparison between original and enhanced triage systems"""
    from LLMs_Practise.LLMs_Practise.TriageSystem import MedicalTriageSystem

    print("Starting comparative triage system evaluation...")

    # Initialize systems
    original_system = MedicalTriageSystem()
    enhanced_system = EnhancedMedicalTriageSystem()

    results = {
        'cases': [],
        'summary': {
            'total_cases': 0,
            'agreement_rate': 0,
            'handoff_difference': 0
        }
    }

    # Process test cases
    case_count = 0
    agreement_count = 0
    handoff_diff_count = 0

    for filename in os.listdir(data_dir):
        if filename.startswith('3000') and filename.endswith('.txt') and case_count < 5:  # Limit to 5 cases for demo
            case_id = filename.replace('.txt', '')
            file_path = os.path.join(data_dir, filename)

            print(f"\nProcessing comparison for case {case_id}...")

            # Read case file
            case_text = read_case_file(file_path)
            if not case_text:
                continue

            # Process with both systems
            try:
                case_data = original_system.parse_medical_case(case_text)

                # Original system
                original_esi = original_system.determine_esi_level(case_data)
                original_recs = original_system.generate_recommendations(original_esi, case_data)

                # Enhanced system
                enhanced_esi = enhanced_system.determine_esi_level(case_data)
                enhanced_recs = enhanced_system.generate_recommendations(enhanced_esi, case_data)

                # Record comparison
                comparison = {
                    'case_id': case_id,
                    'chief_complaint': case_data.get('chief_complaint', '')[:100],
                    'original': {
                        'esi_level': original_esi['esi_level'],
                        'confidence': round(original_esi['confidence'], 2),
                        'needs_handoff': original_esi['needs_handoff']
                    },
                    'enhanced': {
                        'esi_level': enhanced_esi['esi_level'],
                        'confidence': round(enhanced_esi['confidence'], 2),
                        'needs_handoff': enhanced_esi['needs_handoff'],
                        'handoff_analysis': enhanced_esi.get('handoff_analysis', {}),
                        'possible_diagnoses': enhanced_esi.get('possible_diagnoses', [])[:3]
                    }
                }

                # Check agreement
                if original_esi['esi_level'] == enhanced_esi['esi_level']:
                    agreement_count += 1
                    comparison['agreement'] = True
                else:
                    comparison['agreement'] = False

                # Check handoff difference
                if original_esi['needs_handoff'] != enhanced_esi['needs_handoff']:
                    handoff_diff_count += 1
                    comparison['handoff_difference'] = True
                else:
                    comparison['handoff_difference'] = False

                results['cases'].append(comparison)
                case_count += 1

                # Print comparison
                print(f"Chief Complaint: {comparison['chief_complaint']}")
                print(
                    f"Original System: ESI {comparison['original']['esi_level']} | Handoff: {comparison['original']['needs_handoff']}")
                print(
                    f"Enhanced System: ESI {comparison['enhanced']['esi_level']} | Handoff: {comparison['enhanced']['needs_handoff']}")

                if not comparison['agreement']:
                    print("⚠️ ESI LEVEL DIFFERENCE DETECTED")

                if comparison['handoff_difference']:
                    print("⚠️ HANDOFF DECISION DIFFERENCE DETECTED")
                    if enhanced_esi['needs_handoff']:
                        print(
                            f"Handoff reason: {enhanced_esi.get('handoff_analysis', {}).get('primary_reason', 'Unspecified')}")

                print("\nPossible Diagnoses (Enhanced):")
                for diag in enhanced_esi.get('possible_diagnoses', [])[:3]:
                    print(f"- {diag.get('diagnosis', '')}: {diag.get('likelihood', 'medium')} likelihood")

            except Exception as e:
                print(f"Error in comparison for case {case_id}: {str(e)}")

    # Compute summary
    if case_count > 0:
        results['summary'] = {
            'total_cases': case_count,
            'agreement_rate': agreement_count / case_count,
            'handoff_difference_rate': handoff_diff_count / case_count
        }

        print("\n===== Comparison Summary =====")
        print(f"Total Cases: {case_count}")
        print(f"ESI Level Agreement Rate: {results['summary']['agreement_rate'] * 100:.1f}%")
        print(f"Handoff Decision Difference Rate: {results['summary']['handoff_difference_rate'] * 100:.1f}%")

    # Save report
    output_file = output_path or f"triage_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nComparative evaluation report saved to {output_file}")


# Run the evaluation
if __name__ == "__main__":
    data_dir = "D:\\WPI Assignments\\RA-Shraga\\patient_record_example\\patient_record_example"

    # Uncomment the type of evaluation you want to run:

    # Full enhanced evaluation
    evaluator = run_enhanced_triage_evaluation(data_dir)

    # Comparison between original and enhanced systems
    # run_comparative_evaluation(data_dir)