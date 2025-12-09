import argparse
from pathlib import Path
from triage_system import TriageSystem
from confidence_analyzer import ConfidenceAnalyzer
from extended_confidence_analyzer import ExtendedConfidenceAnalyzer
from utils import parse_xml_case, get_xml_files, ensure_output_dirs
from evaluation_pipeline import TriageEvaluator
import json


import sys
sys.path.insert(0, "./site-packages")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Medical Triage Confidence Analysis')
    parser.add_argument('--data-dir', default='../data/annotated_cases',help='Directory containing XML case files')
    print(f"Data directory: {parser.parse_args().data_dir}")
    parser.add_argument('--max-cases', type=int, default=10, help='Maximum number of cases to analyze')
    args = parser.parse_args()
    
    # Create output directories
    ensure_output_dirs()
    
    # Initialize systems
    triage_system = TriageSystem()
    # analyzer = ConfidenceAnalyzer(triage_system)
    analyzer = ExtendedConfidenceAnalyzer(triage_system, output_dir="output/visualizations")
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
        # conf_result = analyzer.analyze_case(case_data, parsed_case['annotations'])
        conf_result = analyzer.analyze_case_with_elicitation(case_data, parsed_case['annotations'])
        conf_result['case_id'] = case_id
        analyzer.results.append(conf_result)
        
        # print("Generating comprehensive visualizations...")
        # analyzer.generate_comprehensive_analysis()
        # print(f"Advanced visualizations saved to output/visualizations/")
        
        print("Generating confidence elicitation visualizations...")
        analyzer.generate_all_visualizations(
            {case_result.get('case_id', f'case_{i}'): case_result 
            for i, case_result in enumerate(analyzer.results)},
            "TrigageSystem",  # Model name
            "MedicalCases"    # Dataset name
        )

      # Generate plot for the incremental analysis part of the results
        if "incremental" in conf_result and "progression" in conf_result["incremental"]:
            plot_path = analyzer.plot_confidence_progression(case_id, conf_result["incremental"])
            print(f"Confidence progression plot saved to {plot_path}")
        else:
            print(f"Warning: Could not generate confidence progression plot for case {case_id}")
        
        # Generate plot
        # plot_path = analyzer.plot_confidence_progression(case_id, conf_result)
        
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