# simplified_extended_confidence_analyzer.py
from confidence_analyzer import ConfidenceAnalyzer
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from eval.prompting_strategies import PromptingStrategies
from eval.sampling_strategies import SamplingStrategies
from eval.aggregation_strategies import AggregationStrategies

class ExtendedConfidenceAnalyzer(ConfidenceAnalyzer):
    """
    Extends ConfidenceAnalyzer with confidence elicitation capabilities from the paper
    "Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs"
    """
    
    def __init__(self, triage_system, output_dir="output"):
        super().__init__(triage_system, output_dir)
        
        # Create additional output directories
        self.confidence_elicitation_dir = os.path.join(output_dir, "confidence_elicitation")
        self.ensure_directories()
    
    def ensure_directories(self):
        """Create additional output directories for confidence elicitation."""
        # Strategies
        strategies = ["vanilla", "cot", "self_probing", "multi_step", "top_k", 
                     "self_random", "prompt_paraphrasing", "misleading",
                     "consistency", "avg_conf", "pair_rank"]
        
        # Visualization types
        vis_types = ["distribution_plots", "performance_tables", "error_analysis"]
        
        # Create main directory
        os.makedirs(self.confidence_elicitation_dir, exist_ok=True)
        
        # Create visualization directories
        for vis_type in vis_types:
            vis_dir = os.path.join(self.confidence_elicitation_dir, vis_type)
            os.makedirs(vis_dir, exist_ok=True)
            
            # Create strategy subdirectories
            for strategy in strategies:
                os.makedirs(os.path.join(vis_dir, strategy), exist_ok=True)
    
    def elicit_confidence(self, case_data, prompt_strategy, 
                         sampling_strategy, aggregation_strategy,
                         n_samples=5, temperature=0.7):
        """
        Elicit confidence using specified strategies.
        
        Args:
            case_data: Medical case data
            prompt_strategy: Strategy for prompting
            sampling_strategy: Strategy for sampling
            aggregation_strategy: Strategy for aggregation
            n_samples: Number of samples
            temperature: Temperature for generation
            
        Returns:
            Dictionary with ESI level, confidence, and metadata
        """
        # Get appropriate prompt function
        if prompt_strategy == "vanilla":
            prompt_func = PromptingStrategies.vanilla_prompt
        elif prompt_strategy == "cot":
            prompt_func = PromptingStrategies.cot_prompt
        elif prompt_strategy == "self_probing":
            # For self-probing, first get preliminary ESI
            vanilla_prompt = PromptingStrategies.vanilla_prompt(case_data)
            vanilla_response = self.triage_system.run(vanilla_prompt)
            
            preliminary_esi, _ = PromptingStrategies.parse_response(vanilla_response, "vanilla") or (3, 0)
            prompt_func = lambda c: PromptingStrategies.self_probing_prompt(c, preliminary_esi)
        elif prompt_strategy == "multi_step":
            prompt_func = PromptingStrategies.multi_step_prompt
        elif prompt_strategy == "top_k":
            prompt_func = PromptingStrategies.top_k_prompt
        else:
            raise ValueError(f"Unknown prompt strategy: {prompt_strategy}")
        
        # Sample responses
        if sampling_strategy == "self_random":
            prompt = prompt_func(case_data)
            responses = SamplingStrategies.self_random_sampling(
                prompt, self.triage_system, n_samples, temperature
            )
        elif sampling_strategy == "misleading":
            responses = SamplingStrategies.misleading_sampling(
                case_data, prompt_func, self.triage_system, n_samples
            )
        elif sampling_strategy == "prompt_paraphrasing":
            responses = SamplingStrategies.prompt_paraphrasing(
                case_data, prompt_func, self.triage_system, n_samples
            )
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
        
        # Parse responses
        parsed_data = []
        for response in responses:
            result = PromptingStrategies.parse_response(response, prompt_strategy)
            if result is not None:
                parsed_data.append(result)
        
        # Special handling for different prompt types
        if prompt_strategy in ["vanilla", "cot"]:
            esi_levels = [data[0] for data in parsed_data if len(data) >= 2]
            confidences = [data[1] for data in parsed_data if len(data) >= 2]
            
            if not esi_levels:
                return {"esi_level": 3, "confidence": 0, "needs_handoff": True}
                
        elif prompt_strategy == "self_probing":
            # For self-probing, we use the preliminary ESI with the parsed confidence
            esi_levels = [preliminary_esi] * len(parsed_data)
            confidences = parsed_data  # Each parsed result is just a confidence value
            
            if not confidences:
                return {"esi_level": preliminary_esi, "confidence": 0, "needs_handoff": True}
                
        elif prompt_strategy == "multi_step":
            # Extract ESI levels, confidences, and step confidences
            esi_levels = [data[0] for data in parsed_data if len(data) >= 3]
            confidences = [data[1] for data in parsed_data if len(data) >= 3]
            step_confidences = [data[2] for data in parsed_data if len(data) >= 3]
            
            if not esi_levels:
                return {"esi_level": 3, "confidence": 0, "needs_handoff": True}
                
        elif prompt_strategy == "top_k":
            # For top-k, each parsed result is (esi_list, conf_list)
            ranked_esi_levels = [(data[0], data[1]) for data in parsed_data if len(data) >= 2]
            
            if not ranked_esi_levels:
                return {"esi_level": 3, "confidence": 0, "needs_handoff": True}
        
        # Aggregate results based on strategy
        if aggregation_strategy == "consistency":
            if prompt_strategy == "top_k":
                # Extract first choice from each ranking
                first_choices = [data[0][0] for data in parsed_data if data[0]]
                esi_level, confidence = AggregationStrategies.consistency_aggregation(first_choices)
            else:
                esi_level, confidence = AggregationStrategies.consistency_aggregation(esi_levels)
                
        elif aggregation_strategy == "avg_conf":
            if prompt_strategy == "top_k":
                # Extract first choice and confidence from each ranking
                first_choices = [data[0][0] for data in parsed_data if data[0]]
                first_confidences = [data[1][0] for data in parsed_data if data[1]]
                esi_level, confidence = AggregationStrategies.avg_conf_aggregation(first_choices, first_confidences)
            else:
                esi_level, confidence = AggregationStrategies.avg_conf_aggregation(esi_levels, confidences)
                
        elif aggregation_strategy == "pair_rank":
            if prompt_strategy == "top_k":
                esi_level, confidence = AggregationStrategies.pair_rank_aggregation(ranked_esi_levels)
            else:
                # Pair-rank is designed for top-k, but we can adapt it
                if prompt_strategy == "multi_step":
                    # Use the step confidences to create a ranking
                    ranked_data = []
                    for i, (esi, conf, steps) in enumerate(zip(esi_levels, confidences, step_confidences)):
                        ranked_data.append(([esi], [conf]))
                    esi_level, confidence = AggregationStrategies.pair_rank_aggregation(ranked_data)
                else:
                    # Create artificial rankings based on confidence
                    ranked_data = []
                    for i, (esi, conf) in enumerate(zip(esi_levels, confidences)):
                        ranked_data.append(([esi], [conf]))
                    esi_level, confidence = AggregationStrategies.pair_rank_aggregation(ranked_data)
        else:
            raise ValueError(f"Unknown aggregation strategy: {aggregation_strategy}")
        
        # Determine if handoff is needed (similar to existing logic)
        needs_handoff = esi_level <= 2 or confidence < 0.6
        
        return {
            "esi_level": esi_level,  # Default to ESI 3 if None
            "confidence": confidence,
            "explanation": responses[0] if responses else "",
            "needs_handoff": needs_handoff,
            "responses": responses,
            "parsed_data": parsed_data,
            "prompt_strategy": prompt_strategy,
            "sampling_strategy": sampling_strategy,
            "aggregation_strategy": aggregation_strategy
        }
    
    def analyze_case_with_elicitation(self, case_data, annotations, strategies=None):
        """
        Analyze a case using both incremental questions and confidence elicitation strategies.
        
        Args:
            case_data: Medical case data
            annotations: Case annotations
            strategies: List of strategy combinations to try, each a tuple of
                        (prompt_strategy, sampling_strategy, aggregation_strategy)
                        
        Returns:
            Dictionary with results from both analyses
        """
        # First do the standard incremental question analysis
        incremental_results = self.analyze_case(case_data, annotations)
        
        # Define default strategies if none provided
        if not strategies:
            strategies = [
                # Vanilla
                ("vanilla", "self_random", "consistency"),
                ("vanilla", "self_random", "avg_conf"),
                ("vanilla", "misleading", "avg_conf"),
                ("vanilla", "prompt_paraphrasing", "consistency"),
                # CoT
                ("cot", "self_random", "consistency"),
                ("cot", "self_random", "avg_conf"),
                ("cot", "prompt_paraphrasing", "consistency"),
                # Self-probing (confidence only)
                ("self_probing", "self_random", "consistency"),
                ("self_probing", "self_random", "avg_conf"),
                # Multi-step
                ("multi_step", "self_random", "avg_conf"),
                # Top-k rankings
                ("top_k", "self_random", "pair_rank"),
                ("top_k", "self_random", "consistency")
            ]
        
        # Run confidence elicitation for each strategy
        elicitation_results = {}
        for prompt_s, sampling_s, agg_s in strategies:
            strategy_key = f"{prompt_s}_{sampling_s}_{agg_s}"
            try:
                result = self.elicit_confidence(
                    case_data, prompt_s, sampling_s, agg_s
                )
                elicitation_results[strategy_key] = result
            except Exception as e:
                print(f"Error with strategy {strategy_key}: {e}")
        
        # Get ground truth ESI level
        esi_annotations = annotations.get('ESI', [])
        ground_truth = None
        if esi_annotations and 'esi_level' in esi_annotations[0]:
            ground_truth = esi_annotations[0]['esi_level']
        
        # Combine results
        return {
            "incremental": incremental_results,
            "elicitation": elicitation_results,
            "ground_truth": ground_truth
        }
    
    def plot_confidence_distribution(self, results, model_name, dataset_name, strategy):
        """Plot distribution of confidence scores for a strategy."""
        # Extract ESI levels and confidences
        esi_levels = []
        confidences = []
        correctness = []
        
        for case_id, case_result in results.items():
            if strategy in case_result.get("elicitation", {}):
                elicitation_result = case_result["elicitation"][strategy]
                ground_truth = case_result.get("ground_truth")
                
                esi_level = elicitation_result.get("esi_level")
                confidence = elicitation_result.get("confidence")
                
                if esi_level is not None and confidence is not None and ground_truth is not None:
                    esi_levels.append(esi_level)
                    confidences.append(confidence)
                    correctness.append(esi_level == ground_truth)
        
        if not confidences:
            print(f"No data available for {strategy}")
            return
        
        # Create distribution plot
        plt.figure(figsize=(10, 6))
        
        # Convert to numpy arrays
        confidences = np.array(confidences)
        correctness = np.array(correctness)
        
        # Separate confidences for correct and incorrect answers
        correct_conf = confidences[correctness]
        incorrect_conf = confidences[~correctness]
        
        # Plot histograms
        plt.hist(correct_conf, bins=10, alpha=0.7, label='Correct ESI', 
                color='blue', range=(0, 100))
        plt.hist(incorrect_conf, bins=10, alpha=0.7, label='Incorrect ESI', 
                color='red', range=(0, 100))
        
        # Add labels and title
        plt.xlabel('Confidence (%)')
        plt.ylabel('Count')
        plt.title(f'Confidence Distribution - {model_name} on {dataset_name} with {strategy}')
        plt.legend()
        
        # Calculate metrics
        accuracy = np.mean(correctness) * 100
        avg_conf = np.mean(confidences)
        ece = self.calculate_ece(confidences/100, correctness)
        
        # Custom simplified AUROC calculation
        sorted_indices = np.argsort(confidences)[::-1]  # Sort in descending order
        sorted_correctness = correctness[sorted_indices]
        
        total_positive = np.sum(correctness)
        total_negative = len(correctness) - total_positive
        
        if total_positive > 0 and total_negative > 0:
            # Count correct rankings
            correct_rankings = 0
            for i in range(len(sorted_correctness)):
                if sorted_correctness[i]:  # If this prediction is correct
                    # Count how many incorrect predictions are ranked below it
                    correct_rankings += np.sum(~sorted_correctness[i+1:])
            
            # Calculate AUROC
            auroc = correct_rankings / (total_positive * total_negative)
        else:
            auroc = 0.5  # Default for degenerate cases
        
        plt.text(0.02, 0.95, f'ACC {accuracy:.1f} / AUROC {auroc:.2f} / ECE {ece:.2f}',
                transform=plt.gca().transAxes, fontsize=10)
        
        # Save figure
        output_dir = os.path.join(self.confidence_elicitation_dir, "distribution_plots", 
                                strategy)  # Use prompt strategy for organization
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{model_name}_{dataset_name}_{strategy}_distribution.png"
        plt.savefig(os.path.join(output_dir, output_file))
        plt.close()
    
    def plot_error_analysis(self, results, model_name, dataset_name, strategy):
        """Plot error analysis for a strategy."""
        # Extract confidences and correctness
        confidences = []
        correctness = []
        
        for case_id, case_result in results.items():
            if strategy in case_result.get("elicitation", {}):
                elicitation_result = case_result["elicitation"][strategy]
                ground_truth = case_result.get("ground_truth")
                
                esi_level = elicitation_result.get("esi_level")
                confidence = elicitation_result.get("confidence")
                
                if esi_level is not None and confidence is not None and ground_truth is not None:
                    confidences.append(confidence)
                    correctness.append(esi_level == ground_truth)
        
        if not confidences:
            print(f"No data available for {strategy}")
            return
        
        plt.figure(figsize=(12, 10))
        
        # Convert to numpy arrays
        confidences = np.array(confidences)
        correctness = np.array(correctness)
        
        # First subplot: Reliability diagram
        plt.subplot(2, 1, 1)
        
        # Create confidence bins
        bin_edges = np.linspace(0, 100, 11)
        bin_indices = np.digitize(confidences, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(bin_edges)-2)
        
        # Calculate accuracy per bin
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(len(bin_edges)-1):
            bin_mask = (bin_indices == i)
            if np.sum(bin_mask) > 0:
                bin_acc = np.mean(correctness[bin_mask])
                bin_conf = (bin_edges[i] + bin_edges[i+1]) / 2 / 100  # Convert to 0-1 scale
                bin_count = np.sum(bin_mask)
                
                bin_accuracies.append(bin_acc)
                bin_confidences.append(bin_conf)
                bin_counts.append(bin_count)
        
        # Plot reliability diagram
        plt.plot(bin_confidences, bin_accuracies, 's-', label='Calibration curve')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        
        for i, (x, y, count) in enumerate(zip(bin_confidences, bin_accuracies, bin_counts)):
            plt.text(x, y + 0.02, str(count), ha='center', va='bottom', fontsize=9)
        
        plt.xlabel('Predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title(f'Reliability Diagram - {model_name} on {dataset_name} with {strategy}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Second subplot: Error rate vs confidence
        plt.subplot(2, 1, 2)
        
        # Calculate error rate per bin
        error_rates = [1 - acc for acc in bin_accuracies]
        conf_bins = [conf * 100 for conf in bin_confidences]  # Convert back to 0-100 scale
        
        # Plot error rate vs confidence
        plt.bar(conf_bins, error_rates, width=9, alpha=0.7)
        
        # Add count labels
        for i, (x, y, count) in enumerate(zip(conf_bins, error_rates, bin_counts)):
            plt.text(x, y + 0.01, str(count), ha='center', va='bottom')
        
        plt.xlabel('Confidence (%)')
        plt.ylabel('Error Rate')
        plt.title('Error Rate vs Confidence')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        
        # Save figure
        output_dir = os.path.join(self.confidence_elicitation_dir, "error_analysis", 
                                strategy.split('_')[0])
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{model_name}_{dataset_name}_{strategy}_error_analysis.png"
        plt.savefig(os.path.join(output_dir, output_file))
        plt.close()
    
    def create_performance_table(self, results, strategies, model_name, dataset_name):
        """Create performance comparison table for different strategies."""
        # Define metrics to track
        metrics = ['accuracy', 'ece', 'auroc', 'auprc_pos', 'auprc_neg']
        
        # Initialize DataFrame
        index = pd.MultiIndex.from_product([metrics, strategies], 
                                         names=['Metric', 'Strategy'])
        table_data = pd.DataFrame(index=index, columns=[dataset_name])
        
        # Calculate metrics for each strategy
        for strategy in strategies:
            # Extract data
            confidences = []
            correctness = []
            
            for case_id, case_result in results.items():
                if strategy in case_result.get("elicitation", {}):
                    elicitation_result = case_result["elicitation"][strategy]
                    ground_truth = case_result.get("ground_truth")
                    
                    esi_level = elicitation_result.get("esi_level")
                    confidence = elicitation_result.get("confidence")
                    
                    if esi_level is not None and confidence is not None and ground_truth is not None:
                        confidences.append(confidence)
                        correctness.append(esi_level == ground_truth)
            
            if confidences:
                # Convert to numpy arrays
                confidences = np.array(confidences)
                correctness = np.array(correctness)
                
                # Calculate metrics
                accuracy = np.mean(correctness)
                ece = self.calculate_ece(confidences/100, correctness)
                
                # Custom simplified AUROC calculation
                sorted_indices = np.argsort(confidences)[::-1]  # Sort in descending order
                sorted_correctness = correctness[sorted_indices]
                
                total_positive = np.sum(correctness)
                total_negative = len(correctness) - total_positive
                
                if total_positive > 0 and total_negative > 0:
                    # Count correct rankings
                    correct_rankings = 0
                    for i in range(len(sorted_correctness)):
                        if sorted_correctness[i]:  # If this prediction is correct
                            # Count how many incorrect predictions are ranked below it
                            correct_rankings += np.sum(~sorted_correctness[i+1:])
                    
                    # Calculate AUROC
                    auroc = correct_rankings / (total_positive * total_negative)
                else:
                    auroc = 0.5  # Default for degenerate cases
                
                # Simplified AUPRC calculations (these are approximations)
                precision_pos = total_positive / len(correctness) if len(correctness) > 0 else 0
                precision_neg = total_negative / len(correctness) if len(correctness) > 0 else 0
                
                # Store metrics
                table_data.loc[('accuracy', strategy), dataset_name] = accuracy
                table_data.loc[('ece', strategy), dataset_name] = ece
                table_data.loc[('auroc', strategy), dataset_name] = auroc
                table_data.loc[('auprc_pos', strategy), dataset_name] = precision_pos
                table_data.loc[('auprc_neg', strategy), dataset_name] = precision_neg
        
        # Save as CSV and return
        output_dir = os.path.join(self.confidence_elicitation_dir, "performance_tables")
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{model_name}_{dataset_name}_performance.csv"
        table_data.to_csv(os.path.join(output_dir, output_file))
        
        return table_data
    
    def calculate_ece(self, confidences, correctness, n_bins=10):
        """Calculate Expected Calibration Error."""
        confidences = np.array(confidences)
        correctness = np.array(correctness)
        
        bin_indices = np.digitize(confidences, np.linspace(0, 1, n_bins+1))
        
        ece = 0
        total_samples = len(confidences)
        
        for bin_idx in range(1, n_bins+1):
            bin_mask = (bin_indices == bin_idx)
            if not np.any(bin_mask):
                continue
                
            bin_count = np.sum(bin_mask)
            bin_confidence = np.mean(confidences[bin_mask])
            bin_accuracy = np.mean(correctness[bin_mask])
            
            ece += (bin_count / total_samples) * np.abs(bin_confidence - bin_accuracy)
        
        return ece
    
    def generate_all_visualizations(self, results, model_name, dataset_name):
        """Generate all visualizations for the results."""
        # Extract all unique strategies
        all_strategies = set()
        for case_result in results.values():
            for strategy in case_result.get("elicitation", {}):
                all_strategies.add(strategy)
        
        strategies = list(all_strategies)
        
        # Generate visualizations for each strategy
        for strategy in strategies:
            self.plot_confidence_distribution(results, model_name, dataset_name, strategy)
            self.plot_error_analysis(results, model_name, dataset_name, strategy)
        
        # Create performance comparison table
        self.create_performance_table(results, strategies, model_name, dataset_name)
        
        # Also plot the standard confidence progression
        for case_id, case_result in results.items():
            if "incremental" in case_result and "progression" in case_result["incremental"]:
                self.plot_esi_confidence_progression(case_id, case_result["incremental"])
