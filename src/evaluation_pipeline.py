import json


class TriageEvaluator:
    def __init__(self, triage_system):
        self.triage_system = triage_system
        self.results = []
        self.metrics = {}
        
    def evaluate_case(self, case_id, parsed_case, case_data):
        """Evaluate a single case against ground truth"""
        # Get annotations
        annotations = parsed_case['annotations']
        
        # Get ground truth
        ground_truth = None
        esi_annotations = annotations.get('ESI', [])
        if esi_annotations and 'esi_level' in esi_annotations[0]:
            ground_truth = esi_annotations[0]['esi_level']
        
        # Get prediction with full information
        prediction = self.triage_system.determine_esi_level(case_data)
        
        # Record result
        result = {
            'case_id': case_id,
            'chief_complaint': case_data.get('chief_complaint', ''),
            'predicted': prediction,
            'ground_truth': ground_truth,
            'correct': prediction['esi_level'] == ground_truth if ground_truth else None,
            'arrival_method': case_data.get('visit_info', {}).get('Arrival Transport', '')
        }
        
        self.results.append(result)
        return result
    
    def calculate_metrics(self):
        """Calculate evaluation metrics"""
        if not self.results:
            return {}

        # Only compute accuracy on cases with ground truth
        gt_cases = [r for r in self.results if r.get('ground_truth') is not None]
        if gt_cases:
            correct_cases = [r for r in gt_cases if r['correct'] is True]
            accuracy = len(correct_cases) / len(gt_cases)
            within_one = [r for r in gt_cases if abs(r['predicted']['esi_level'] - r['ground_truth']) <= 1]
            accuracy_within_one = len(within_one) / len(gt_cases)
        else:
            accuracy = None
            accuracy_within_one = None

        # ESI distribution
        esi_distribution = {}
        for level in range(1, 6):
            count = sum(1 for r in self.results if r['predicted']['esi_level'] == level)
            esi_distribution[level] = count / len(self.results)
        
        # Handoff rate
        handoff_rate = sum(1 for r in self.results if r['predicted']['needs_handoff']) / len(self.results)
        
        self.metrics = {
            'total_cases': len(self.results),
            'cases_with_ground_truth': len(gt_cases),
            'accuracy': accuracy,
            'accuracy_within_one': accuracy_within_one,
            'esi_distribution': esi_distribution,
            'handoff_rate': handoff_rate
        }
        
        return self.metrics
    
    def generate_report(self, output_path="output/reports/evaluation_report.json"):
        """Generate and save evaluation report"""
        if not self.metrics:
            self.calculate_metrics()
        
        report = {
            'metrics': self.metrics,
            'case_results': self.results
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return output_path
