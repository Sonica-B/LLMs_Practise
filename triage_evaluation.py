import json
from enhanced_triage_system import EnhancedMedicalTriageSystem


class TriageEvaluationSystem:
    def __init__(self, triage_system=None):
        self.triage_system = triage_system or EnhancedMedicalTriageSystem()
        self.evaluation_results = {
            'cases': [],
            'metrics': {
                'overall': {},
                'by_esi_level': {},
                'by_arrival_method': {},
                'handoff_analysis': {}
            }
        }
        self.confusion_matrix = [[0 for _ in range(5)] for _ in range(5)]  # 5x5 for ESI levels 1-5

    def evaluate_case(self, case_id, case_text, ground_truth=None):
        """Evaluate a single case with detailed metrics"""
        try:
            # Parse and process the case
            case_data = self.triage_system.parse_medical_case(case_text)
            esi_result = self.triage_system.determine_esi_level(case_data)
            recommendations = self.triage_system.generate_recommendations(esi_result, case_data)

            # Extract original acuity if available
            original_acuity = None
            try:
                original_acuity = int(case_data.get('visit_info', {}).get('Acuity', ''))
            except:
                pass

            # Record result
            result = {
                'case_id': case_id,
                'chief_complaint': case_data.get('chief_complaint', ''),
                'predicted': {
                    'esi_level': esi_result['esi_level'],
                    'confidence': round(esi_result['confidence'], 2),
                    'needs_handoff': esi_result['needs_handoff'],
                    'handoff_analysis': esi_result.get('handoff_analysis', {}),
                    'possible_diagnoses': esi_result.get('possible_diagnoses', []),
                    'risk_factors': esi_result.get('risk_factors', [])
                },
                'actual': {
                    'esi_level': original_acuity,
                    'disposition': case_data.get('visit_info', {}).get('Disposition', 'Not specified'),
                    'arrival_method': case_data.get('visit_info', {}).get('Arrival Transport', 'Not specified')
                },
                'recommendations': {
                    'wait_time': recommendations['estimated_wait_time'],
                    'structured_recommendations': recommendations.get('structured_recommendations', {})
                }
            }

            # Update confusion matrix if we have ground truth
            if original_acuity and 1 <= original_acuity <= 5:
                pred_idx = esi_result['esi_level'] - 1
                actual_idx = original_acuity - 1
                self.confusion_matrix[actual_idx][pred_idx] += 1

            # Store result
            self.evaluation_results['cases'].append(result)
            return result

        except Exception as e:
            print(f"Error evaluating case {case_id}: {str(e)}")
            return None

    def generate_metrics(self):
        """Generate comprehensive evaluation metrics"""
        cases = self.evaluation_results['cases']
        metrics = {}

        if not cases:
            return {"error": "No cases evaluated"}

        # Overall metrics
        metrics['overall'] = {
            'total_cases': len(cases),
            'average_confidence': round(sum(c['predicted']['confidence'] for c in cases) / len(cases), 2),
            'handoff_rate': round(sum(1 for c in cases if c['predicted']['needs_handoff']) / len(cases), 2),
            'esi_distribution': {
                level: sum(1 for c in cases if c['predicted']['esi_level'] == level) / len(cases)
                for level in range(1, 6)
            }
        }

        # Accuracy metrics (if we have ground truth)
        ground_truth_cases = [c for c in cases if c['actual']['esi_level'] is not None]
        if ground_truth_cases:
            correct = sum(1 for c in ground_truth_cases if c['predicted']['esi_level'] == c['actual']['esi_level'])
            metrics['accuracy'] = {
                'overall': round(correct / len(ground_truth_cases), 2),
                'within_one_level': round(sum(1 for c in ground_truth_cases
                                              if abs(c['predicted']['esi_level'] - c['actual']['esi_level']) <= 1)
                                          / len(ground_truth_cases), 2),
                'confusion_matrix': self.confusion_matrix
            }

        # Handoff analysis
        metrics['handoff_analysis'] = self.analyze_handoff_decisions()

        # By arrival method
        arrival_methods = set(c['actual']['arrival_method'] for c in cases)
        metrics['by_arrival_method'] = {
            method: {
                'count': sum(1 for c in cases if c['actual']['arrival_method'] == method),
                'handoff_rate': round(sum(1 for c in cases
                                          if
                                          c['actual']['arrival_method'] == method and c['predicted']['needs_handoff']) /
                                      max(1, sum(1 for c in cases if c['actual']['arrival_method'] == method)), 2),
                'average_esi': round(sum(c['predicted']['esi_level'] for c in cases
                                         if c['actual']['arrival_method'] == method) /
                                     max(1, sum(1 for c in cases if c['actual']['arrival_method'] == method)), 2)
            }
            for method in arrival_methods
        }

        # Most common risk factors
        all_risk_factors = []
        for case in cases:
            all_risk_factors.extend(case['predicted']['risk_factors'])

        risk_factor_counts = {}
        for factor in all_risk_factors:
            risk_factor_counts[factor] = risk_factor_counts.get(factor, 0) + 1

        metrics['common_risk_factors'] = {
            factor: count / len(cases)
            for factor, count in sorted(risk_factor_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }

        self.evaluation_results['metrics'] = metrics
        return metrics

    def analyze_handoff_decisions(self):
        """Analyze the distribution and effectiveness of handoff decisions"""
        cases = self.evaluation_results['cases']

        # Skip if no cases
        if not cases:
            return {"error": "No cases to analyze"}

        # Calculate handoff rate by ESI level
        handoff_by_esi = {}
        for level in range(1, 6):
            level_cases = [c for c in cases if c['predicted']['esi_level'] == level]
            if level_cases:
                handoff_rate = sum(1 for c in level_cases if c['predicted']['needs_handoff']) / len(level_cases)
                handoff_by_esi[str(level)] = {
                    'count': len(level_cases),
                    'handoff_rate': round(handoff_rate, 2),
                    'avg_confidence': round(sum(c['predicted']['confidence'] for c in level_cases) / len(level_cases),
                                            2)
                }

        # Calculate handoff reasons distribution
        reasons = {}
        handoff_cases = [c for c in cases if c['predicted']['needs_handoff']]
        for case in handoff_cases:
            reason = case['predicted']['handoff_analysis'].get('primary_reason', 'Unspecified')
            reasons[reason] = reasons.get(reason, 0) + 1

        reason_distribution = {reason: round(count / max(1, len(handoff_cases)), 2)
                               for reason, count in reasons.items()}

        return {
            'total_cases': len(cases),
            'total_handoffs': len(handoff_cases),
            'overall_handoff_rate': round(len(handoff_cases) / len(cases), 2) if cases else 0,
            'handoff_by_esi': handoff_by_esi,
            'reason_distribution': reason_distribution
        }

    def print_evaluation_summary(self, case_result=None):
        """Print evaluation summary for a case or overall metrics"""
        if case_result:
            print("\n===== Case Evaluation Summary =====")
            print(f"Case ID: {case_result['case_id']}")
            print(f"Chief Complaint: {case_result['chief_complaint'][:100]}...")
            print(
                f"\nPredicted ESI Level: {case_result['predicted']['esi_level']} (Confidence: {case_result['predicted']['confidence']})")

            if case_result['actual']['esi_level']:
                print(f"Actual ESI Level: {case_result['actual']['esi_level']}")

            print(f"\nHandoff Required: {case_result['predicted']['needs_handoff']}")
            if case_result['predicted']['needs_handoff']:
                print(
                    f"Primary Reason: {case_result['predicted']['handoff_analysis'].get('primary_reason', 'Unspecified')}")
                print("\nHandoff Reasons:")
                reasons = case_result['predicted']['handoff_analysis'].get('reasons', [])
                if reasons:
                    for reason in reasons:
                        if isinstance(reason, dict):
                            print(f"- {reason.get('category', '')}: {reason.get('explanation', '')}")
                        else:
                            print(f"- {reason}")
            else:
                print("Case can be handled autonomously by AI system")

            print("\nTop Possible Diagnoses:")
            for diag in case_result['predicted']['possible_diagnoses'][:3]:
                print(f"- {diag.get('diagnosis', '')}: {diag.get('likelihood', 'medium')} likelihood")

            print("\nKey Risk Factors:")
            for factor in case_result['predicted']['risk_factors'][:5]:
                print(f"- {factor}")

            print("\nRecommended Tests:")
            for lab in case_result['recommendations'].get('structured_recommendations', {}).get('labs', [])[:3]:
                if isinstance(lab, dict):
                    print(f"- {lab.get('name', '')}: {lab.get('rationale', '')}")
                else:
                    print(f"- {lab}")

            print(f"\nEstimated Wait Time: {case_result['recommendations']['wait_time']} hours")
        else:
            # Print overall metrics if available
            if self.evaluation_results.get('metrics'):
                metrics = self.evaluation_results['metrics']
                print("\n===== Overall Evaluation Metrics =====")
                print(f"Total Cases: {metrics['overall']['total_cases']}")
                print(f"Average Confidence: {metrics['overall']['average_confidence']}")
                print(f"Handoff Rate: {metrics['overall']['handoff_rate'] * 100:.1f}%")

                print("\nESI Level Distribution:")
                for level, percent in metrics['overall']['esi_distribution'].items():
                    print(f"ESI {level}: {percent * 100:.1f}%")

                if 'accuracy' in metrics:
                    print(f"\nAccuracy: {metrics['accuracy']['overall'] * 100:.1f}%")
                    print(f"Accuracy within one level: {metrics['accuracy']['within_one_level'] * 100:.1f}%")

                if 'handoff_analysis' in metrics:
                    handoff = metrics['handoff_analysis']
                    print(f"\nOverall Handoff Rate: {handoff['overall_handoff_rate'] * 100:.1f}%")

                    print("\nHandoff Rate by ESI Level:")
                    for level, data in handoff['handoff_by_esi'].items():
                        print(f"ESI {level}: {data['handoff_rate'] * 100:.1f}% of {data['count']} cases")

                    print("\nTop Handoff Reasons:")
                    for reason, percent in sorted(handoff['reason_distribution'].items(),
                                                  key=lambda x: x[1], reverse=True)[:3]:
                        print(f"- {reason}: {percent * 100:.1f}%")

                print("\nMost Common Risk Factors:")
                for factor, percent in list(metrics['common_risk_factors'].items())[:5]:
                    print(f"- {factor}: {percent * 100:.1f}%")

    def generate_detailed_report(self, output_path="triage_evaluation_report.json"):
        """Generate and save detailed evaluation report"""
        if not self.evaluation_results.get('metrics'):
            self.generate_metrics()

        with open(output_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)

        print(f"\nDetailed evaluation report saved to {output_path}")