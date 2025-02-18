from langchain_ollama import OllamaLLM
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
import re


class MedicalTriageSystem:
    def __init__(self):
        # Initialize OllamaLLM
        self.llm = OllamaLLM(model="mistral")

        # Define risk conditions and thresholds
        self.HIGH_RISK_CONDITIONS = [
            "chest pain", "shortness of breath", "confusion",
            "altered mental status", "severe pain", "fall",
            "pneumonia", "encephalopathy"
        ]

        self.CRITICAL_VITALS = {
            'systolic': (90, 180),
            'diastolic': (60, 110),
            'heart_rate': (60, 100),
            'respiratory_rate': (12, 20),
            'temperature': (36.5, 38.5),
            'o2_saturation': 92
        }

        self.ARRIVAL_RISK_SCORES = {
            'AMBULANCE': 2,
            'WALK IN': 0,
            'WHEELCHAIR': 1
        }

    def parse_medical_case(self, case_text):
        """Parse structured medical case data"""
        sections = {}
        current_section = ""

        for line in case_text.split('\n'):
            if '====' in line:
                section_name = line.strip('=').strip()
                current_section = section_name
                sections[current_section] = []
            elif current_section and line.strip():
                sections[current_section].append(line.strip())

        # Extract structured Q&A pairs
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
            for line in sections['ER Visit Info']:
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

    def extract_vitals(self, text):
        """Extract vital signs from text"""
        vitals = {}

        patterns = {
            'systolic': r'(?:BP|blood pressure)[^\d]*(\d+)/\d+',
            'diastolic': r'(?:BP|blood pressure)[^\d]*\d+/(\d+)',
            'heart_rate': r'(?:HR|heart rate|pulse)[^\d]*(\d+)',
            'respiratory_rate': r'(?:RR|respiratory rate)[^\d]*(\d+)',
            'temperature': r'(?:T|temp)[^\d]*(\d+\.?\d*)',
            'o2_saturation': r'(?:O2|oxygen|sat)[^\d]*(\d+)%?'
        }

        for vital, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    vitals[vital] = float(match.group(1))
                except ValueError:
                    continue

        return vitals

    def analyze_risk_factors(self, case_data):
        """Analyze risk factors from medical case data"""
        risk_score = 0
        risk_factors = []

        # Check chief complaint and summary for high-risk conditions
        text_to_check = f"{case_data['chief_complaint']} {case_data['summary']}"
        for condition in self.HIGH_RISK_CONDITIONS:
            if condition.lower() in text_to_check.lower():
                risk_score += 2
                risk_factors.append(f"High-risk condition: {condition}")

        # Analyze vital signs
        vitals = self.extract_vitals(case_data['summary'])
        for vital, value in vitals.items():
            if vital in ['systolic', 'diastolic']:
                min_val, max_val = self.CRITICAL_VITALS[vital]
                if value < min_val or value > max_val:
                    risk_score += 1
                    risk_factors.append(f"Abnormal {vital}: {value}")

        # Check arrival method
        arrival = case_data['visit_info'].get('Arrival Transport', '')
        risk_score += self.ARRIVAL_RISK_SCORES.get(arrival, 0)
        if arrival == 'AMBULANCE':
            risk_factors.append("Ambulance arrival")

        # Check original acuity
        try:
            acuity = float(case_data['visit_info'].get('Acuity', 5))
            risk_score += (5 - acuity) * 1.5
            if acuity <= 2:
                risk_factors.append(f"High acuity: {acuity}")
        except ValueError:
            pass

        return risk_score, risk_factors

    def determine_esi_level(self, case_data):
        """Determine ESI level based on case analysis"""
        risk_score, risk_factors = self.analyze_risk_factors(case_data)

        prompt = PromptTemplate(
            template="""Based on the following patient information, determine the appropriate ESI level (1-5):

            Chief Complaint: {chief_complaint}

            Summary: {summary}

            Risk Score: {risk_score}
            Risk Factors: {risk_factors}

            Key Q&A Information:
            {qa_summary}

            Guidelines:
            ESI 1: Immediate life-saving intervention required
            ESI 2: High risk situation, confused/lethargic/disoriented, or severe pain/distress
            ESI 3: Multiple resources needed, but vital signs stable
            ESI 4: One resource needed
            ESI 5: No resources needed

            Provide ESI level (1-5) and detailed explanation:""",
            input_variables=["chief_complaint", "summary", "risk_score",
                             "risk_factors", "qa_summary"]
        )

        # Prepare Q&A summary focusing on critical information
        qa_summary = self._prepare_qa_summary(case_data['qa_pairs'])

        # Get LLM prediction
        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(
            chief_complaint=case_data['chief_complaint'],
            summary=case_data['summary'][:500],  # Limit length
            risk_score=risk_score,
            risk_factors="\n".join(risk_factors),
            qa_summary=qa_summary
        )

        # Extract ESI level
        try:
            esi_level = int(re.search(r'ESI (\d)', response).group(1))
        except:
            # Default based on risk score if parsing fails
            esi_level = self._default_esi_from_risk(risk_score)

        confidence = self.calculate_confidence(response, case_data, risk_score)

        return {
            'esi_level': esi_level,
            'confidence': confidence,
            'explanation': response,
            'risk_factors': risk_factors,
            'risk_score': risk_score,
            'needs_handoff': confidence < 0.85 or esi_level <= 2
        }

    def _prepare_qa_summary(self, qa_pairs):
        """Prepare relevant Q&A summary"""
        critical_keywords = ['pain', 'breathing', 'consciousness', 'injury',
                             'symptoms', 'medication', 'medical history']

        relevant_qa = []
        for qa in qa_pairs:
            if any(keyword in qa['question'].lower() for keyword in critical_keywords):
                relevant_qa.append(f"Q: {qa['question']}\nA: {qa['answer']}")

        return "\n".join(relevant_qa[:5])  # Limit to most relevant 5 pairs

    def _default_esi_from_risk(self, risk_score):
        """Determine default ESI level based on risk score"""
        if risk_score >= 8:
            return 1
        elif risk_score >= 6:
            return 2
        elif risk_score >= 4:
            return 3
        elif risk_score >= 2:
            return 4
        return 5

    def calculate_confidence(self, response, case_data, risk_score):
        """Calculate confidence score for ESI prediction"""
        base_confidence = 0.8

        # Adjust based on risk score
        if risk_score > 6:
            base_confidence *= 0.9  # Reduce confidence for high-risk cases

        # Adjust based on data completeness
        if len(case_data['qa_pairs']) < 5:
            base_confidence *= 0.95

        # Adjust based on vital signs presence
        vitals = self.extract_vitals(case_data['summary'])
        if len(vitals) < 3:
            base_confidence *= 0.95

        # Adjust based on arrival method
        if case_data['visit_info'].get('Arrival Transport') == 'AMBULANCE':
            base_confidence *= 0.9

        return min(base_confidence, 1.0)

    def generate_recommendations(self, esi_result, case_data):
        """Generate medical recommendations based on ESI assessment"""
        needs_handoff = esi_result['needs_handoff']

        prompt = PromptTemplate(
            template="""Based on:
            ESI Level: {esi_level}
            Risk Factors: {risk_factors}
            Chief Complaint: {chief_complaint}
            Current Summary: {summary}

            Generate:
            1. Immediate medical staff actions required
            2. Recommended diagnostic tests/labs
            3. Required monitoring level
            4. Estimated resource requirements""",
            input_variables=["esi_level", "risk_factors", "chief_complaint", "summary"]
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)
        recommendations = chain.run(
            esi_level=esi_result['esi_level'],
            risk_factors="\n".join(esi_result['risk_factors']),
            chief_complaint=case_data['chief_complaint'],
            summary=case_data['summary'][:500]
        )

        return {
            'staff_recommendations': recommendations,
            'estimated_wait_time': self._calculate_wait_time(
                esi_result['esi_level'],
                esi_result['risk_score']
            ),
            'handoff_required': needs_handoff
        }

    def _calculate_wait_time(self, esi_level, risk_score):
        """Calculate estimated wait time based on ESI level and risk score"""
        base_times = {
            1: 0,  # Immediate
            2: 0.25,  # 15 minutes
            3: 1,  # 1 hour
            4: 2,  # 2 hours
            5: 3  # 3 hours
        }

        wait_time = base_times.get(esi_level, 2)

        # Adjust based on risk score
        if risk_score > 6:
            wait_time *= 0.5  # Reduce wait time for high-risk patients

        return max(0, min(wait_time, 4))  # Cap between 0 and 4 hours