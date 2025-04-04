import re

class PromptingStrategies:
    """
    Implements various prompting strategies for confidence elicitation as described in the paper:
    "Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs"
    """
    
    @staticmethod
    def vanilla_prompt(question, is_multi_choice=False):
        """
        Basic prompt asking for an answer and confidence level.
        """
        if is_multi_choice:
            prompt = f"""Read the question, provide your answer and your confidence in this answer.
Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
```Answer and Confidence (0-100): [ONLY the option letter; not a complete sentence], [Your confidence level, please only include the numerical number in the range of 0-100]%```

Only the answer and confidence, don't give me the explanation.
Question: {question}

Now, please answer this question and provide your confidence level."""
        else:
            prompt = f"""Read the question, provide your answer and your confidence in this answer.
Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
```Answer and Confidence (0-100): [ONLY the number; not a complete sentence], [Your confidence level, please only include the numerical number in the range of 0-100]%```

Only the answer and confidence, don't give me the explanation.
Question: {question}

Now, please answer this question and provide your confidence level."""
        return prompt
    
    @staticmethod
    def cot_prompt(question, is_multi_choice=False):
        """
        Chain-of-Thought prompt asking for step-by-step reasoning before answer and confidence.
        """
        if is_multi_choice:
            prompt = f"""Read the question, analyze step by step, provide your answer and your confidence in this answer.
Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
```Explanation: [insert step-by-step analysis here]
Answer and Confidence (0-100): [ONLY the option letter; not a complete sentence], [Your confidence level, please only include the numerical number in the range of 0-100]%```

Only give me the reply according to this format, don't give me any other words.
Question: {question}

Now, please answer this question and provide your confidence level. Let's think it step by step."""
        else:
            prompt = f"""Read the question, analyze step by step, provide your answer and your confidence in this answer.
Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
```Explanation: [insert step-by-step analysis here]
Answer and Confidence (0-100): [ONLY the number; not a complete sentence], [Your confidence level, please only include the numerical number in the range of 0-100]%```

Only give me the reply according to this format, don't give me any other words.
Question: {question}

Now, please answer this question and provide your confidence level. Let's think it step by step."""
        return prompt
    
    @staticmethod
    def self_probing_prompt(question, answer_candidate):
        """
        Self-probing prompt that asks how likely a given answer is correct.
        """
        prompt = f"""Question: {question}
Possible Answer: {answer_candidate}

Q: How likely is the above answer to be correct? Please first show your reasoning concisely and then answer with the following format:
```Confidence: [the probability of answer {answer_candidate} to be correct, not the one you think correct, please only include the numerical number]```"""
        return prompt
    
    @staticmethod
    def multi_step_prompt(question):
        """
        Multi-step prompt that breaks down the problem with confidence in each step.
        """
        prompt = f"""Read the question, break down the problem into K steps, think step by step, give your confidence in each step, and then derive your final answer and your confidence in this answer.
Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
```Step 1: [Your reasoning], Confidence: [ONLY the confidence value that this step is correct]%
...
Step K: [Your reasoning], Confidence: [ONLY the confidence value that this step is correct]%
Final Answer and Overall Confidence (0-100): [ONLY the answer type; not a complete sentence], [Your confidence value]%```

Question: {question}"""
        return prompt
    
    @staticmethod
    def top_k_prompt(question, k=3):
        """
        Top-K prompt that asks for k best guesses with associated confidence.
        """
        prompt = f"""Provide your {k} best guesses and the probability that each is correct (0% to 100%) for the following question. Give ONLY the task output description of your guesses and probabilities, no other words or explanation. For example:

G1: <ONLY the task output description of first most likely guess; not a complete sentence, just the guess!> P1: <ONLY the probability that G1 is correct, without any extra commentary whatsoever; just the probability!>
...
G{k}: <ONLY the task output description of {k}-th most likely guess> P{k}: <ONLY the probability that G{k} is correct, without any extra commentary whatsoever; just the probability!>

Question: {question}"""
        return prompt
    
    @staticmethod
    def parse_vanilla_response(response):
        """Parse the response from vanilla prompt to extract answer and confidence."""
        pattern = r"Answer and Confidence.*?:.*?([^,]+),\s*(\d+)%"
        match = re.search(pattern, response, re.DOTALL)
        
        if match:
            answer = match.group(1).strip()
            confidence = int(match.group(2))
            return answer, confidence
        return None, None
    
    @staticmethod
    def parse_cot_response(response):
        """Parse the response from CoT prompt to extract answer and confidence."""
        pattern = r"Answer and Confidence.*?:.*?([^,]+),\s*(\d+)%"
        match = re.search(pattern, response, re.DOTALL)
        
        if match:
            answer = match.group(1).strip()
            confidence = int(match.group(2))
            return answer, confidence
        return None, None
    
    @staticmethod
    def parse_self_probing_response(response):
        """Parse the response from self-probing prompt to extract confidence."""
        pattern = r"Confidence:\s*(\d+)"
        match = re.search(pattern, response)
        
        if match:
            confidence = int(match.group(1))
            return confidence
        return None
    
    @staticmethod
    def parse_multi_step_response(response):
        """Parse the response from multi-step prompt to extract step confidences and final answer."""
        # Extract step confidences
        step_pattern = r"Step \d+:.*?Confidence:\s*(\d+)%"
        step_confidences = [int(conf) for conf in re.findall(step_pattern, response)]
        
        # Extract final answer and confidence
        final_pattern = r"Final Answer and Overall Confidence.*?:.*?([^,]+),\s*(\d+)%"
        final_match = re.search(final_pattern, response, re.DOTALL)
        
        if final_match:
            answer = final_match.group(1).strip()
            confidence = int(final_match.group(2))
            return answer, confidence, step_confidences
        return None, None, step_confidences
    
    @staticmethod
    def parse_top_k_response(response, k=3):
        """Parse the response from top-k prompt to extract k guesses and confidences."""
        guesses = []
        confidences = []
        
        # Match all G#: guess P#: confidence patterns
        pattern = r"G\d+:\s*([^\n]+)\s*P\d+:\s*(\d+)%?"
        matches = re.findall(pattern, response)
        
        if matches:
            for match in matches[:k]:  # Limit to k matches
                guess = match[0].strip()
                confidence = int(match[1])
                guesses.append(guess)
                confidences.append(confidence)
        
        return guesses, confidences