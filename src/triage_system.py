import re
import json
import requests
import logging

class TriageSystem:
    def __init__(self, api_base="http://localhost:11434"):
        # Ollama API base URL
        self.api_base = api_base
        
        # Define risk conditions
        self.HIGH_RISK_CONDITIONS = [
            "chest pain", "shortness of breath", "confusion",
            "altered mental status", "severe pain"
        ]
    
    def run(self, prompt, temperature=0.7):
        """Run the LLM with the given prompt"""
        url = f"{self.api_base}/api/generate"
        payload = {
            "model": "mistral",
            "prompt": prompt,
            "temperature": temperature
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code != 200:
            raise RuntimeError(f"Ollama request failed ({response.status_code}): {response.text}")

        # Parse streaming response
        full_text = ""
        for line in response.text.splitlines():
            if not line:
                continue
            try:
                data = json.loads(line)
                full_text += data.get("response", "")
            except Exception:
                continue

        if not full_text.strip():
            raise RuntimeError("Ollama returned an empty response.")

        return full_text

    def _parse_json_response(self, text):
        """Try to extract a JSON object from free-form LLM output."""
        try:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end+1])
        except Exception:
            return None
        return None
    
    def parse_medical_case(self, case_text):
        """Parse medical case from raw text"""
        sections = {}
        current_section = ""
        
        for line in case_text.split('\n'):
            if '====' in line:
                section_name = line.strip('=').strip()
                current_section = section_name
                sections[current_section] = []
            elif current_section and line.strip():
                sections[current_section].append(line.strip())
        
        # Extract QA pairs
        qa_pairs = []
        if 'Question Answer Pair' in sections:
            qa_text = '\n'.join(sections['Question Answer Pair'])
            qa_blocks = re.findall(r'\d+\.\s*- Question \d+: (.*?)\s*- Answer \d+: (.*?)(?=\d+\.|$)',
                                  qa_text, re.DOTALL)
            qa_pairs = [{'question': q.strip(), 'answer': a.strip()}
                        for q, a in qa_blocks]
        
        # Parse ER visit info
        visit_info = {}
        if 'ER Visit Info' in sections:
            for line in sections.get('ER Visit Info', []):
                if ':' in line:
                    key, value = line.split(':', 1)
                    visit_info[key.strip()] = value.strip()
        
        return {
            'summary': '\n'.join(sections.get('Summary of Current ER Visit', [])),
            'chief_complaint': '\n'.join(sections.get('Patient Chief Complaint', [])),
            'history': '\n'.join(sections.get('Patient History Summary', [])),
            'qa_pairs': qa_pairs,
            'visit_info': visit_info
        }
    
    def determine_esi_level(self, case_data, qa_count=None):
        """Determine ESI level with confidence tracking"""
        # Limit QA pairs if specified
        if qa_count is not None:
            case_data = case_data.copy()
            case_data['qa_pairs'] = case_data['qa_pairs'][:qa_count]
        
        # Analyze risk factors
        risk_factors = self.identify_risk_factors(case_data)
        
        # Prepare prompt for LLM (request JSON for robust parsing)
        prompt = f"""Determine the appropriate Emergency Severity Index (ESI) level (1-5) for this patient.
Return ONLY JSON with keys: esi_level (1-5), confidence (0-100), handoff ("yes"/"no"), explanation (short text).

            Chief Complaint: {case_data['chief_complaint']}
            Summary: {case_data['summary'][:500]}
            Risk Factors: {"\n".join(risk_factors)}
            
            Q&A Information:
            {"\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in case_data.get('qa_pairs', [])])}

            Guidelines:
            ESI 1: Immediate life-saving intervention required
            ESI 2: High risk situation or severe pain/distress
            ESI 3: Multiple resources needed, vital signs stable
            ESI 4: One resource needed
            ESI 5: No resources needed"""
        
        # Get LLM prediction
        response = self.run(prompt)

        # Try JSON parse first
        parsed = self._parse_json_response(response)
        esi_level = None
        confidence = None
        needs_handoff = None

        if isinstance(parsed, dict):
            try:
                esi_level = int(parsed.get("esi_level"))
            except Exception:
                esi_level = None
            try:
                conf_val = float(str(parsed.get("confidence")).replace("%",""))
                confidence = conf_val/100.0 if conf_val > 1 else conf_val
            except Exception:
                confidence = None
            handoff_val = str(parsed.get("handoff", "")).lower()
            if handoff_val in ("yes","no"):
                needs_handoff = handoff_val == "yes"
        
        # Fallback to regex if JSON parsing failed
        if esi_level is None:
            esi_match = re.search(r'ESI(?:\\s*level)?[^0-9]*([1-5])', response, re.IGNORECASE)
            if esi_match:
                esi_level = int(esi_match.group(1))

        if confidence is None:
            conf_match = re.search(r'Confidence(?:\\s*level)?[^0-9]*([0-9]+(?:\\.\\d+)?)%?', response, re.IGNORECASE)
            if conf_match:
                raw_conf = float(conf_match.group(1))
                confidence = raw_conf / 100.0 if raw_conf > 1 else raw_conf

        if needs_handoff is None:
            handoff_match = re.search(r'handoff.*?(yes|no)', response, re.IGNORECASE)
            needs_handoff = True if handoff_match and handoff_match.group(1).lower() == 'yes' else False

        # Final fallbacks to avoid aborting a case
        if esi_level is None:
            logging.warning("Falling back to default ESI 3 for unparsable response")
            esi_level = 3
        if confidence is None:
            logging.warning("Falling back to default confidence 0.5 for unparsable response")
            confidence = 0.5
        if needs_handoff is None:
            needs_handoff = esi_level <= 2
        
        return {
            'esi_level': esi_level,
            'confidence': confidence,
            'explanation': response,
            'risk_factors': risk_factors,
            'needs_handoff': needs_handoff,
            'qa_count': len(case_data.get('qa_pairs', []))
        }
    
    def predict_resource_needs(self, case_data):
        """Predict the number of resources needed"""
        resources = 0
        combined_text = f"{case_data['chief_complaint']} {case_data['summary']}"
        
        # Check for lab needs
        if any(term in combined_text.lower() for term in ["blood", "test", "lab"]):
            resources += 1
        
        # Check for imaging needs
        if any(term in combined_text.lower() for term in ["xray", "ct", "scan"]):
            resources += 1
        
        return resources

    def check_concerning_vitals(self, case_data):
        """Check for concerning vital signs"""
        summary = case_data.get('summary', '').lower()
        return any(term in summary for term in ["tachycardia", "hypotension", "hypertension", "fever"])

    def identify_risk_factors(self, case_data):
        """Identify risk factors in case data"""
        risk_factors = []
        text_to_check = f"{case_data['chief_complaint']} {case_data['summary']}"
        
        # Check for high-risk conditions
        for condition in self.HIGH_RISK_CONDITIONS:
            if condition.lower() in text_to_check.lower():
                risk_factors.append(f"High-risk condition: {condition}")
        
        # Check arrival method
        arrival = case_data['visit_info'].get('Arrival Transport', '')
        if arrival == 'AMBULANCE':
            risk_factors.append("Ambulance arrival")
        
        return risk_factors
