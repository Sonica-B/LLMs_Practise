import numpy as np
import math
from scipy.optimize import minimize

class AggregationStrategies:
    """
    Implements various aggregation strategies for combining multiple responses.
    """
    
    @staticmethod
    def consistency_aggregation(answers, original_answer=None):
        """
        Measure the degree of agreement among candidate outputs.
        
        Args:
            answers: List of candidate answers
            original_answer: Original answer (if available)
            
        Returns:
            Aggregated confidence based on consistency
        """
        if not answers:
            return 0.0
            
        # If original answer is provided, use it as reference
        reference = original_answer if original_answer is not None else answers[0]
        
        # Count matches
        matches = sum(1 for answer in answers if answer == reference)
        
        # Compute consistency
        consistency = matches / len(answers)
        
        return consistency
    
    @staticmethod
    def avg_conf_aggregation(answers, confidences, original_answer=None):
        """
        Average verbalized confidences weighted by matches.
        
        Args:
            answers: List of candidate answers
            confidences: List of verbalized confidences for each answer
            original_answer: Original answer (if available)
            
        Returns:
            Aggregated confidence based on avg-conf method
        """
        if not answers or not confidences or len(answers) != len(confidences):
            return 0.0
            
        # If original answer is provided, use it as reference
        reference = original_answer if original_answer is not None else answers[0]
        
        # Sum confidences of matching answers
        matching_conf_sum = sum(conf for ans, conf in zip(answers, confidences) if ans == reference)
        total_conf_sum = sum(confidences)
        
        # Compute average confidence
        if total_conf_sum == 0:
            return 0.0
            
        avg_conf = matching_conf_sum / total_conf_sum
        
        return avg_conf
    
    @staticmethod
    def pair_rank_aggregation(top_k_responses):
        """
        Use ranking information from top-k responses to estimate confidence.
        
        Args:
            top_k_responses: List of (guesses, confidences) tuples from top-k prompts
            
        Returns:
            Dictionary mapping answers to their estimated probabilities
        """
        # Extract all unique answers
        all_answers = set()
        for guesses, _ in top_k_responses:
            all_answers.update(guesses)
        
        # Create a mapping from answer to index
        answer_to_idx = {answer: i for i, answer in enumerate(all_answers)}
        n_answers = len(all_answers)
        
        # Initialize parameters for softmax (one per answer)
        initial_params = np.zeros(n_answers)
        
        # Define the loss function (negative log-likelihood)
        def loss_func(params):
            # Convert to probabilities using softmax
            exp_params = np.exp(params - np.max(params))  # Subtract max for numerical stability
            probs = exp_params / np.sum(exp_params)
            
            total_loss = 0.0
            
            # For each top-k response
            for guesses, _ in top_k_responses:
                # For each pair of answers in the ranking
                for i in range(len(guesses)):
                    for j in range(i+1, len(guesses)):
                        # Get the answers and their indices
                        answer_i = guesses[i]
                        answer_j = guesses[j]
                        idx_i = answer_to_idx[answer_i]
                        idx_j = answer_to_idx[answer_j]
                        
                        # Probability of ranking answer_i above answer_j
                        p_i = probs[idx_i]
                        p_j = probs[idx_j]
                        
                        # Avoid division by zero
                        if p_i + p_j > 0:
                            p_i_given_i_or_j = p_i / (p_i + p_j)
                            # Add negative log-likelihood
                            total_loss -= math.log(p_i_given_i_or_j)
            
            return total_loss
        
        # Optimize the parameters
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(np.exp(x)) - 1.0}
        result = minimize(loss_func, initial_params, method='SLSQP', 
                         constraints=[constraints], options={'disp': False})
        
        # Convert optimal parameters to probabilities
        optimal_params = result.x
        exp_params = np.exp(optimal_params - np.max(optimal_params))
        probs = exp_params / np.sum(exp_params)
        
        # Create mapping from answer to probability
        answer_probs = {answer: probs[answer_to_idx[answer]] for answer in all_answers}
        
        return answer_probs