# sampling_strategies.py
import random

class SamplingStrategies:
    """Implements sampling strategies for generating multiple responses."""
    
    @staticmethod
    def self_random_sampling(prompt, llm, n_samples=5, temperature=0.7):
        """Sample using model's inherent randomness."""
        responses = []
        for _ in range(n_samples):
            response = llm.run(prompt, temperature=temperature)
            responses.append(response)
        return responses
    
    @staticmethod
    def misleading_sampling(case_data, prompt_func, llm, n_samples=5):
        """Sample by providing misleading risk factors."""
        # Define misleading hints for medical triage
        misleading_hints = [
            "Patient presents with slightly elevated heart rate.",
            "Patient reports mild pain (3/10).",
            "Patient has a history of anxiety.",
            "Patient arrived by private vehicle.",
            "Patient has been waiting for 30 minutes.",
            "Patient's symptoms started 2 days ago.",
            "Patient has no known allergies.",
            "Patient is afebrile."
        ]
        
        # Get baseline response
        base_prompt = prompt_func(case_data)
        base_response = llm.run(base_prompt)
        
        # Generate responses with misleading information
        responses = [base_response]  # Include the baseline response
        
        # Create a copy of case data for augmentation
        for i in range(n_samples - 1):  # -1 because we already have the baseline
            augmented_case = case_data.copy()
            
            # Add misleading information to summary
            hint = random.choice(misleading_hints)
            if 'summary' in augmented_case:
                augmented_case['summary'] = f"{augmented_case['summary']} {hint}"
            else:
                augmented_case['summary'] = hint
                
            # Get response with augmented case
            prompt = prompt_func(augmented_case)
            response = llm.run(prompt)
            responses.append(response)
        
        return responses
    
    @staticmethod
    def prompt_paraphrasing(case_data, prompt_func, llm, n_samples=5):
        """Sample by paraphrasing the case presentation."""
        # Base prompt
        base_prompt = prompt_func(case_data)
        responses = []
        
        # Different phrasings for case presentation
        phrasings = [
            "Read this patient case and determine the ESI level:",
            "Evaluate the following emergency department patient:",
            "Assess this patient's triage level based on the ESI scale:",
            "Review this case and assign an appropriate ESI level:",
            "Determine the correct ESI triage category for this patient:",
            "Based on the Emergency Severity Index, categorize this patient:"
        ]
        
        # Generate responses with different phrasings
        for i in range(min(n_samples, len(phrasings))):
            # Replace the first line of the prompt with a different phrasing
            prompt_lines = base_prompt.split('\n')
            if len(prompt_lines) > 0:
                prompt_lines[0] = phrasings[i]
            modified_prompt = '\n'.join(prompt_lines)
            
            response = llm.run(modified_prompt)
            responses.append(response)
        
        # If we need more samples, use the base prompt for the rest
        while len(responses) < n_samples:
            response = llm.run(base_prompt)
            responses.append(response)
        
        return responses