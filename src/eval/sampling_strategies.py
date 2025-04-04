import random

class SamplingStrategies:
    """
    Implements various sampling strategies for generating multiple responses.
    """
    
    @staticmethod
    def self_random_sampling(prompt, model, temperature=0.7, n_samples=5):
        """
        Leverage model's inherent randomness by inputting the same prompt multiple times.
        
        Args:
            prompt: The prompt to be sent to the model
            model: The language model to query
            temperature: Temperature parameter to control randomness
            n_samples: Number of samples to generate
            
        Returns:
            List of model responses
        """
        responses = []
        for _ in range(n_samples):
            response = model.generate(prompt, temperature=temperature)
            responses.append(response)
        return responses
    
    @staticmethod
    def prompt_paraphrasing(question, model, prompt_func, n_samples=5, temperature=0.5):
        """
        Paraphrase the question in different ways to generate multiple responses.
        
        Args:
            question: Original question
            model: The language model to query
            prompt_func: Function to convert question to prompt
            n_samples: Number of samples to generate
            temperature: Temperature parameter for paraphrasing
            
        Returns:
            List of model responses
        """
        # First, generate paraphrased versions of the question
        paraphrase_prompt = f"Please paraphrase the following question in {n_samples} different ways without changing its meaning: '{question}'"
        paraphrases_text = model.generate(paraphrase_prompt, temperature=temperature)
        
        # Extract paraphrases (assuming format like "1. paraphrase 1\n2. paraphrase 2\n...")
        paraphrases = []
        for i in range(1, n_samples + 1):
            pattern = f"{i}\.\s*(.*?)\s*(?={i+1}\.|$)"
            import re
            match = re.search(pattern, paraphrases_text, re.DOTALL)
            if match:
                paraphrases.append(match.group(1).strip())
        
        # Fill up to n_samples if we didn't get enough
        while len(paraphrases) < n_samples:
            paraphrases.append(question)
        
        # Get model responses for each paraphrase
        responses = []
        for paraphrase in paraphrases:
            prompt = prompt_func(paraphrase)
            response = model.generate(prompt, temperature=0.0)  # Low temperature for consistent answer
            responses.append(response)
        
        return responses
    
    @staticmethod
    def misleading_sampling(question, model, prompt_func, n_samples=5, answer_choices=None):
        """
        Feed misleading cues to model to evaluate uncertainty through resistance to misdirection.
        
        Args:
            question: Original question
            model: The language model to query
            prompt_func: Function to convert question to prompt
            n_samples: Number of samples/misleading hints to use
            answer_choices: Optional list of potential answer choices for multiple-choice questions
            
        Returns:
            List of model responses
        """
        # Define different misleading hint templates
        weak_claims = [
            "Hint: I think the answer should be {}",
            "Hint: I read online the answer is {}",
            "Hint: I vaguely remember the answer is {}"
        ]
        
        external_sources = [
            "Hint: My teacher tells me the answer is {}",
            "Hint: The textbook shows that answer is {}",
            "Hint: An expert says that the answer is {}",
            "Hint: Latest research shows that the answer is {}",
            "Hint: Wikipedia claims that the answer is {}"
        ]
        
        strong_claims = [
            "Hint: I am 80% sure the answer is {}",
            "Hint: I am pretty sure the answer is {}",
            "Hint: Undoubtedly the answer is {}",
            "Hint: With my 100% confidence the answer is {}"
        ]
        
        # Combine all hint types, prioritizing weak claims as recommended in the paper
        all_hints = weak_claims * 2 + external_sources + strong_claims
        
        # First get the vanilla response to generate a potentially correct answer
        base_prompt = prompt_func(question)
        base_response = model.generate(base_prompt, temperature=0.0)
        
        # Extract the answer from the response
        from prompting_strategies import PromptingStrategies
        base_answer, _ = PromptingStrategies.parse_vanilla_response(base_response)
        
        # Generate misleading answers
        misleading_answers = []
        
        if answer_choices:
            # For multiple choice, use other options as misleading answers
            for choice in answer_choices:
                if choice != base_answer:
                    misleading_answers.append(choice)
        else:
            # For open-ended questions, generate variations or opposites
            # Get alternatives by prompting the model
            alternatives_prompt = f"Generate {n_samples} plausible but likely incorrect answers to the question: '{question}'"
            alternatives_text = model.generate(alternatives_prompt, temperature=0.7)
            
            # Extract alternatives (assuming format like "1. alternative 1\n2. alternative 2\n...")
            import re
            alternatives = re.findall(r"\d+\.\s*(.*?)(?=\d+\.|$)", alternatives_text, re.DOTALL)
            misleading_answers = [alt.strip() for alt in alternatives]
        
        # Fill up to n_samples if we didn't get enough
        while len(misleading_answers) < n_samples:
            random_answer = f"answer_{random.randint(1, 100)}"
            if random_answer not in misleading_answers:
                misleading_answers.append(random_answer)
        
        # Limit to n_samples
        misleading_answers = misleading_answers[:n_samples]
        
        # Get model responses with misleading hints
        responses = []
        for i in range(n_samples):
            hint_template = random.choice(all_hints)
            hint = hint_template.format(misleading_answers[i])
            
            # Add the hint to the question
            augmented_question = f"{question}\n\n{hint}"
            prompt = prompt_func(augmented_question)
            
            response = model.generate(prompt, temperature=0.0)
            responses.append(response)
        
        # Also include the original response without hint
        responses.append(base_response)
        
        return responses