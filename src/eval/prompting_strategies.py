# prompting_strategies.py
import re

class PromptingStrategies:
    """
    Implements various prompting strategies for confidence elicitation in medical triage.
    """
    
    @staticmethod
    def vanilla_prompt(case_data):
        """Basic prompt asking for ESI level and confidence."""
        prompt = f"""Read the following patient information and determine the appropriate Emergency Severity Index (ESI) level (1-5):

Chief Complaint: {case_data.get('chief_complaint', '')}
Summary: {case_data.get('summary', '')}
Patient History: {case_data.get('history', '')}

Provide your ESI level assessment (1-5) and your confidence level (0-100%) in this format:
ESI Level: [number]
Confidence: [number]%
Brief explanation: [your reasoning]
"""
        return prompt
    
    @staticmethod
    def cot_prompt(case_data):
        """Chain-of-Thought prompt for ESI determination with confidence."""
        prompt = f"""Read the following patient information and determine the appropriate Emergency Severity Index (ESI) level (1-5):

Chief Complaint: {case_data.get('chief_complaint', '')}
Summary: {case_data.get('summary', '')}
Patient History: {case_data.get('history', '')}

Think step by step about the following factors:
1. Is this patient dying or about to die? (ESI 1)
2. Is this a high-risk situation or severe pain/distress? (ESI 2)
3. How many resources will this patient need? (For ESI 3-5)
4. Are vital signs concerning? (Can affect ESI level)

After your analysis, provide:
ESI Level: [number]
Confidence: [number]%
Step-by-step reasoning: [your detailed analysis]
"""
        return prompt
    
    @staticmethod
    def self_probing_prompt(case_data, preliminary_esi):
        """Self-probing prompt that asks how likely a preliminary ESI assessment is correct."""
        prompt = f"""Review the following patient case:

Chief Complaint: {case_data.get('chief_complaint', '')}
Summary: {case_data.get('summary', '')}
Patient History: {case_data.get('history', '')}

Preliminary ESI Level Assessment: {preliminary_esi}

Q: How likely is the above ESI level assessment to be correct? Analyze the assessment, 
provide your reasoning, and give your confidence in this assessment.

Confidence: [number]%
Reasoning: [your explanation]
"""
        return prompt
    
    @staticmethod
    def multi_step_prompt(case_data):
        """Multi-step prompt for ESI determination with confidence at each step."""
        prompt = f"""Read the following patient information:

Chief Complaint: {case_data.get('chief_complaint', '')}
Summary: {case_data.get('summary', '')}
Patient History: {case_data.get('history', '')}

Break down the ESI level determination into steps and evaluate your confidence in each step:

Step 1: Assess if this is a life-threatening condition requiring immediate intervention (ESI 1).
Confidence in this assessment: [number]%

Step 2: Assess if this is a high-risk situation or involves severe pain/distress (ESI 2).
Confidence in this assessment: [number]%

Step 3: Determine how many resources this patient will need (for ESI 3-5).
Confidence in this assessment: [number]%

Step 4: Evaluate if vital signs are within normal limits (can affect ESI level).
Confidence in this assessment: [number]%

Final ESI Level: [number]
Overall Confidence: [number]%
Reasoning: [your explanation]
"""
        return prompt
    
    @staticmethod
    def top_k_prompt(case_data, k=3):
        """Top-K prompt for ESI determination with ranked confidences."""
        prompt = f"""Review the following patient case:

Chief Complaint: {case_data.get('chief_complaint', '')}
Summary: {case_data.get('summary', '')}
Patient History: {case_data.get('history', '')}

Provide your {k} most likely ESI level assessments and the probability that each is correct (0-100%):

G1: [ESI level] P1: [probability]%
G2: [ESI level] P2: [probability]%
G3: [ESI level] P3: [probability]%

Brief reasoning for your top assessment: [concise explanation]
"""
        return prompt[:k]
    
    @staticmethod
    def parse_response(response, prompt_type="vanilla"):
        """Parse ESI level and confidence from response based on prompt type."""
        if prompt_type in ["vanilla", "cot"]:
            esi_pattern = r"ESI Level:\s*(\d)"
            conf_pattern = r"Confidence:\s*(\d+)"
            
            esi_match = re.search(esi_pattern, response)
            conf_match = re.search(conf_pattern, response)
            
            if esi_match and conf_match:
                esi_level = int(esi_match.group(1))
                confidence = int(conf_match.group(1))
                return esi_level, confidence
                
        elif prompt_type == "self_probing":
            conf_pattern = r"Confidence:\s*(\d+)"
            conf_match = re.search(conf_pattern, response)
            
            if conf_match:
                confidence = int(conf_match.group(1))
                return confidence
                
        elif prompt_type == "multi_step":
            # Extract step confidences
            step_pattern = r"Confidence in this assessment:\s*(\d+)"
            step_confidences = [int(conf) for conf in re.findall(step_pattern, response)]
            
            # Extract final ESI and confidence
            esi_pattern = r"Final ESI Level:\s*(\d)"
            conf_pattern = r"Overall Confidence:\s*(\d+)"
            
            esi_match = re.search(esi_pattern, response)
            conf_match = re.search(conf_pattern, response)
            
            if esi_match and conf_match and step_confidences:
                esi_level = int(esi_match.group(1))
                confidence = int(conf_match.group(1))
                return esi_level, confidence, step_confidences
                
        elif prompt_type == "top_k":
            # Parse ESI levels and confidences
            pattern = r"G\d+:\s*(\d)\s*P\d+:\s*(\d+)"
            matches = re.findall(pattern, response)
            
            if matches:
                esi_levels = [int(m[0]) for m in matches]
                confidences = [int(m[1]) for m in matches]
                return esi_levels, confidences
        
        # Default return if parsing fails
        return None