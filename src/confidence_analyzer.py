import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
from pathlib import Path
import os
from matplotlib.lines import Line2D

class ConfidenceAnalyzer:
    def __init__(self, triage_system, output_dir="output"):
        self.triage_system = triage_system
        self.results = []
        self.output_dir = output_dir
    
    def analyze_case(self, case_data, annotations):
        """Analyze confidence progression with incremental questions"""
        # Get original QA pairs and their relevance
        qa_pairs = case_data.get('qa_pairs', [])
        qa_annotations = annotations.get('QA', [])
        
        # Add relevance information to QA pairs
        for i, qa in enumerate(qa_pairs):
            if i < len(qa_annotations):
                qa['relevance'] = qa_annotations[i].get('relevance', 'Unknown')
        
        # Sort QA pairs by relevance (Essential first)
        relevance_order = {"Essential": 0, "Optional": 1, "Wrong": 2, "Unknown": 3}
        sorted_qa = sorted(qa_pairs, key=lambda x: relevance_order.get(x.get('relevance', 'Unknown'), 4))
        
        # Track confidence progression
        progression = []
        
        # Start with no questions
        case_with_no_qa = case_data.copy()
        case_with_no_qa['qa_pairs'] = []
        
        base_result = self.triage_system.determine_esi_level(case_with_no_qa)
        progression.append({
            'questions_asked': 0,
            'question_type': 'None',
            'confidence': base_result['confidence'],
            'esi_level': base_result['esi_level'],
            'needs_handoff': base_result['needs_handoff']
        })
        
        # Add questions incrementally
        for i in range(1, len(sorted_qa) + 1):
            result = self.triage_system.determine_esi_level(case_data, qa_count=i)
            
            progression.append({
                'questions_asked': i,
                'question_type': sorted_qa[i-1].get('relevance', 'Unknown'),
                'confidence': result['confidence'],
                'esi_level': result['esi_level'],
                'needs_handoff': result['needs_handoff']
            })
        
        # Get ground truth ESI level
        esi_annotations = annotations.get('ESI', [])
        ground_truth = None
        if esi_annotations and 'esi_level' in esi_annotations[0]:
            ground_truth = esi_annotations[0]['esi_level']
        
        return {
            'progression': progression,
            'ground_truth': ground_truth
        }
    
    def plot_confidence_progression(self, case_id, result):
        """Generate plot showing confidence progression"""
        progression = result['progression']
        ground_truth = result['ground_truth']
        
        plt.figure(figsize=(10, 6))
        
        # Plot confidence over questions
        questions = [p['questions_asked'] for p in progression]
        confidence = [p['confidence'] for p in progression]
        esi_levels = [p['esi_level'] for p in progression]
        
        # Color by question type
        colors = []
        for p in progression:
            question_type = p.get('question_type', 'Unknown')
            if question_type == 'Essential':
                colors.append('green')
            elif question_type == 'Optional':
                colors.append('blue')
            elif question_type == 'Wrong':
                colors.append('red')
            else:
                colors.append('gray')
        
        # Plot confidence line
        plt.plot(questions, confidence, 'o-', color='black', linewidth=2)
        
        # Color points by question type
        for i, (q, conf, color) in enumerate(zip(questions, confidence, colors)):
            plt.plot(q, conf, 'o', color=color, markersize=8)
        
        # Add handoff threshold line
        plt.axhline(y=0.75, color='red', linestyle='--', label='Handoff Threshold')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Essential Questions'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Optional Questions'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Wrong Questions'),
            Line2D([0], [0], color='red', linestyle='--', label='Handoff Threshold')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.title(f"Confidence Progression - Case {case_id}")
        plt.xlabel('Number of Questions Asked')
        plt.ylabel('Confidence')
        plt.ylim(0.4, 1.0)
        plt.grid(True, alpha=0.3)
        
        output_path = f"output/confidence_analysis/{case_id}_progression.png"
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def generate_summary_report(self):
        """Generate summary report of confidence analysis"""
        if not self.results:
            return "No results to analyze"
        
        # Calculate summary statistics
        summary = {
            'cases_analyzed': len(self.results),
            'average_initial_confidence': np.mean([r['progression'][0]['confidence'] for r in self.results]),
            'average_final_confidence': np.mean([r['progression'][-1]['confidence'] for r in self.results]),
            'confidence_change': {},
            'question_impact': {}
        }
        
        # Calculate confidence change by question type
        question_types = ['Essential', 'Optional', 'Wrong']
        for q_type in question_types:
            changes = []
            for result in self.results:
                for i in range(1, len(result['progression'])):
                    if result['progression'][i]['question_type'] == q_type:
                        change = result['progression'][i]['confidence'] - result['progression'][i-1]['confidence']
                        changes.append(change)
            
            if changes:
                summary['question_impact'][q_type] = {
                    'average_change': float(np.mean(changes)),
                    'count': len(changes)
                }
        
        # Save summary to JSON
        output_path = "output/reports/confidence_summary.json"
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return output_path
    
    def plot_esi_confidence_progression(self, case_id, result=None):
        """Enhanced plot showing both ESI level and confidence progression"""
        if result is None:
            # Try to find the case in results
            for r in self.results:
                if r.get('case_id') == case_id:
                    result = r
                    break
            if result is None:
                print(f"No data found for case {case_id}")
                return None
                
        progression = result['progression']
        ground_truth = result.get('ground_truth')
        
        # Create a figure with two subplots sharing x-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, 
                                        gridspec_kw={'height_ratios': [1, 1.5]})
        
        # Extract data
        questions = [p['questions_asked'] for p in progression]
        confidence = [p['confidence'] for p in progression]
        esi_levels = [p['esi_level'] for p in progression]
        
        # Get question types for coloring
        colors = []
        question_types = []
        for p in progression:
            q_type = p.get('question_type', 'None')
            question_types.append(q_type)
            if q_type == 'Essential':
                colors.append('green')
            elif q_type == 'Optional':
                colors.append('blue')
            elif q_type == 'Wrong':
                colors.append('red')
            else:
                colors.append('gray')
        
        # Plot ESI level progression (top subplot)
        ax1.plot(questions, esi_levels, 'o-', markersize=10, linewidth=2, color='purple')
        ax1.set_title(f"ESI Level Progression - Case {case_id}", fontsize=14)
        ax1.set_ylabel("ESI Level", fontsize=12)
        ax1.set_ylim(0.5, 5.5)  # ESI levels are 1-5
        ax1.invert_yaxis()  # Invert so ESI 1 (highest acuity) is at the top
        ax1.grid(True, alpha=0.3)
        
        # Add ground truth if available
        if ground_truth:
            ax1.axhline(y=ground_truth, color='green', linestyle='-', alpha=0.5)
            ax1.text(0.02, ground_truth, f"Ground Truth: ESI {ground_truth}", 
                    fontsize=10, va='center', color='green')
        
        # Mark where ESI level changes
        for i in range(1, len(esi_levels)):
            if esi_levels[i] != esi_levels[i-1]:
                ax1.axvline(x=questions[i], color='red', linestyle='--', alpha=0.3)
        
        # Plot confidence progression (bottom subplot)
        ax2.plot(questions, confidence, 'o-', linewidth=2, color='blue')
        
        # Color points by question type
        for i, (q, conf, color) in enumerate(zip(questions, confidence, colors)):
            ax2.plot(q, conf, 'o', color=color, markersize=10)
        
        ax2.set_title("Confidence Progression", fontsize=14)
        ax2.set_ylabel("Confidence", fontsize=12)
        ax2.set_xlabel("Number of Questions Asked", fontsize=12)
        ax2.set_ylim(0.4, 1.0)
        ax2.grid(True, alpha=0.3)
        
        # Add handoff threshold line
        ax2.axhline(y=0.75, color='red', linestyle='--', label='Handoff Threshold')
        
        # Add question type annotations
        for i, (q, conf, q_type) in enumerate(zip(questions, confidence, question_types)):
            if i > 0:  # Skip the first point (no question)
                ax2.annotate(f"Q{i}\n({q_type})", (q, conf), 
                            textcoords="offset points", xytext=(0, 10), 
                            ha='center', fontsize=8)
        
        # Add legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, 
                    label='Essential Question'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, 
                    label='Optional Question'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, 
                    label='Wrong Question'),
            Line2D([0], [0], color='red', linestyle='--', label='Handoff Threshold')
        ]
        ax2.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        vis_dir = f"{self.output_dir}/visualizations"
        os.makedirs(vis_dir, exist_ok=True)
        output_path = f"{vis_dir}/{case_id}_esi_confidence.png"
        plt.savefig(output_path, dpi=300)
        plt.close(fig)
        
        return output_path
    
    def plot_confidence_accuracy_correlation(self):
        """Create a scatter plot showing correlation between confidence and accuracy"""
        if not self.results:
            print("No results available for confidence-accuracy correlation")
            return None
            
        # Gather prediction data
        data = []
        for result in self.results:
            ground_truth = result.get('ground_truth')
            if not ground_truth:
                continue
                
            for p in result['progression']:
                data.append({
                    'case_id': result.get('case_id', ''),
                    'confidence': p['confidence'],
                    'esi_level': p['esi_level'],
                    'ground_truth': ground_truth,
                    'question_count': p['questions_asked'],
                    'question_type': p.get('question_type', 'None'),
                    'correct': p['esi_level'] == ground_truth,
                    'error': abs(p['esi_level'] - ground_truth)
                })
        
        if not data:
            print("No data available for confidence-accuracy correlation")
            return None
            
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(data)
        
        # Create confidence bins
        df['confidence_bin'] = pd.cut(df['confidence'], 
                                     bins=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                     labels=['0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0'])
        
        # Calculate accuracy by confidence bin
        accuracy_by_confidence = df.groupby('confidence_bin').agg(
            accuracy=('correct', 'mean'),
            count=('correct', 'count'),
            avg_confidence=('confidence', 'mean'),
            avg_error=('error', 'mean')
        ).reset_index()
        
        # Plot accuracy vs confidence
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot with point size based on count
        plt.scatter(accuracy_by_confidence['avg_confidence'], 
                  accuracy_by_confidence['accuracy'],
                  s=accuracy_by_confidence['count']*5, 
                  alpha=0.7)
        
        # Add labels for each point
        for i, row in accuracy_by_confidence.iterrows():
            plt.annotate(f"n={row['count']}\nerr={row['avg_error']:.2f}", 
                       (row['avg_confidence'], row['accuracy']),
                       xytext=(5, 5), textcoords='offset points')
        
        # Add diagonal line representing perfect calibration
        plt.plot([0.4, 1.0], [0.4, 1.0], 'k--', alpha=0.5, label='Perfect calibration')
        
        plt.title('Confidence vs. Accuracy Correlation', fontsize=14)
        plt.xlabel('Average Confidence', fontsize=12)
        plt.ylabel('Accuracy (% Correct)', fontsize=12)
        plt.xlim(0.4, 1.0)
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        vis_dir = f"{self.output_dir}/visualizations"
        os.makedirs(vis_dir, exist_ok=True)
        output_path = f"{vis_dir}/confidence_accuracy_correlation.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
    
    def plot_question_type_impact(self):
        """Analyze and visualize the impact of different question types"""
        if not self.results:
            print("No results available for question type impact analysis")
            return None
            
        # Gather data on question impacts
        impacts = []
        for result in self.results:
            progression = result['progression']
            ground_truth = result.get('ground_truth')
            
            for i in range(1, len(progression)):
                # Calculate changes after each question
                confidence_change = progression[i]['confidence'] - progression[i-1]['confidence']
                esi_changed = progression[i]['esi_level'] != progression[i-1]['esi_level']
                
                # Check if change improved accuracy (if ground truth available)
                accuracy_improved = None
                if ground_truth:
                    prev_error = abs(progression[i-1]['esi_level'] - ground_truth)
                    curr_error = abs(progression[i]['esi_level'] - ground_truth)
                    accuracy_improved = curr_error < prev_error
                
                impacts.append({
                    'case_id': result.get('case_id', ''),
                    'question_num': i,
                    'question_type': progression[i].get('question_type', 'Unknown'),
                    'confidence_before': progression[i-1]['confidence'],
                    'confidence_after': progression[i]['confidence'],
                    'confidence_change': confidence_change,
                    'esi_changed': esi_changed,
                    'accuracy_improved': accuracy_improved
                })
        
        if not impacts:
            print("No data available for question type impact analysis")
            return None
            
        # Convert to DataFrame
        impact_df = pd.DataFrame(impacts)
        
        # Analyze confidence change by question type
        q_type_impact = impact_df.groupby('question_type').agg(
            avg_confidence_change=('confidence_change', 'mean'),
            esi_change_rate=('esi_changed', 'mean'),
            accuracy_improvement_rate=('accuracy_improved', lambda x: x.mean() if x.count() > 0 else np.nan),
            count=('question_type', 'count')
        ).reset_index()
        
        # Create a figure with multiple subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # 1. Bar chart of average confidence change by question type
        sns.barplot(x='question_type', y='avg_confidence_change', data=q_type_impact, ax=ax1)
        ax1.set_title('Average Confidence Change by Question Type', fontsize=14)
        ax1.set_xlabel('Question Type', fontsize=12)
        ax1.set_ylabel('Average Confidence Change', fontsize=12)
        
        # Add text labels above bars
        for i, row in q_type_impact.iterrows():
            ax1.text(i, row['avg_confidence_change'], f"n={row['count']}", 
                    ha='center', va='bottom', fontsize=10)
        
        # 2. Impact on ESI and accuracy
        X = np.arange(len(q_type_impact))
        width = 0.35
        
        # ESI change rate bars
        ax2.bar(X - width/2, q_type_impact['esi_change_rate'], width, label='ESI Change Rate')
        
        # Accuracy improvement rate bars (if available)
        if not q_type_impact['accuracy_improvement_rate'].isna().all():
            ax2.bar(X + width/2, q_type_impact['accuracy_improvement_rate'], 
                   width, label='Accuracy Improvement Rate')
        
        ax2.set_title('Impact on ESI & Accuracy by Question Type', fontsize=14)
        ax2.set_xticks(X)
        ax2.set_xticklabels(q_type_impact['question_type'])
        ax2.set_xlabel('Question Type', fontsize=12)
        ax2.set_ylabel('Rate', fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.legend()
        
        # Add count labels
        for i, row in q_type_impact.iterrows():
            ax2.text(i, 0.05, f"n={row['count']}", ha='center', fontsize=10)
        
        plt.tight_layout()
        vis_dir = f"{self.output_dir}/visualizations"
        os.makedirs(vis_dir, exist_ok=True)
        output_path = f"{vis_dir}/question_type_impact.png"
        plt.savefig(output_path, dpi=300)
        plt.close(fig)
        
        return output_path
    
    def generate_comprehensive_analysis(self):
        """Generate all visualizations and return their paths"""
        os.makedirs(f"{self.output_dir}/visualizations", exist_ok=True)
        
        output_files = []
        
        # Plot enhanced progression for all cases
        for result in self.results:
            case_id = result.get('case_id', 'unknown')
            file_path = self.plot_esi_confidence_progression(case_id, result)
            output_files.append(file_path)
        
        # Correlation and impact analyses
        correlation_path = self.plot_confidence_accuracy_correlation()
        if correlation_path:
            output_files.append(correlation_path)
            
        impact_path = self.plot_question_type_impact()
        if impact_path:
            output_files.append(impact_path)
        
        return output_files