import argparse
from pathlib import Path
from triage_system import TriageSystem
from confidence_analyzer import ConfidenceAnalyzer
from utils import parse_xml_case, get_xml_files, ensure_output_dirs
from evaluation_pipeline import TriageEvaluator
import json

import sys
sys.path.insert(0, "./site-packages")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Medical Triage Confidence Analysis')
    parser.add_argument('--data-dir', default='LLMs_Practise/data/annotated_cases', 
                       help='Directory containing XML case files')
    parser.add_argument('--max-cases', type=int, default=10,
                       help='Maximum number of cases to analyze')
    args = parser.parse_args()
    
    # Create output directories
    ensure_output_dirs()
    
    # Initialize systems
    triage_system = TriageSystem()
    analyzer = ConfidenceAnalyzer(triage_system)
    evaluator = TriageEvaluator(triage_system)
    

    data_dir = Path(args.data_dir)
    print(f"Looking for files in: {data_dir.absolute()}")
    all_files = list(data_dir.glob('*'))
    print(f"All files in directory: {[f.name for f in all_files]}")
    

    # Get XML files
    xml_files = get_xml_files(args.data_dir, args.max_cases)
    print(f"Found {len(xml_files)} XML files to analyze")
    
    # Process each case
    for i, xml_file in enumerate(xml_files):
        case_id = xml_file.stem
        print(f"Processing case {i+1}/{len(xml_files)}: {case_id}")
        
        # Parse XML case
        parsed_case = parse_xml_case(xml_file)
        
        # Parse medical case
        case_data = triage_system.parse_medical_case(parsed_case['case_text'])
        
        # Evaluate against ground truth
        eval_result = evaluator.evaluate_case(case_id, parsed_case, case_data)
        
        # Analyze confidence progression
        conf_result = analyzer.analyze_case(case_data, parsed_case['annotations'])
        conf_result['case_id'] = case_id
        analyzer.results.append(conf_result)
        
        print("Generating comprehensive visualizations...")
        analyzer.generate_comprehensive_analysis()
        print(f"Advanced visualizations saved to output/visualizations/")

        # Generate plot
        plot_path = analyzer.plot_confidence_progression(case_id, conf_result)
        
        # Save results
        with open(f"output/reports/{case_id}_detailed.json", 'w') as f:
            json.dump({**eval_result, 'confidence_analysis': conf_result}, f, indent=2)
    
    # Generate evaluation report
    eval_path = evaluator.generate_report()
    print(f"Evaluation complete. Report saved to {eval_path}")
    
    # Generate confidence analysis report
    conf_path = analyzer.generate_summary_report()
    print(f"Confidence analysis complete. Summary saved to {conf_path}")


if __name__ == "__main__":
    main()