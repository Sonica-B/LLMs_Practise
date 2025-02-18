from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import re
import json


class MedicalTriageSystem:
    def __init__(self):
        self.llm = Ollama(model="mistral")
        self.embeddings = HuggingFaceEmbeddings()
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory="./medical_triage_db"
        )

        # ESI-specific thresholds
        self.HIGH_RISK_CONDITIONS = [
            "chest pain", "shortness of breath", "stroke",
            "severe pain", "altered mental status"
        ]
        self.CRITICAL_VITALS = {
            'systolic': (90, 180),  # (min, max)
            'diastolic': (60, 110),
            'heart_rate': (60, 100),
            'respiratory_rate': (12, 20),
            'temperature': (36.5, 38.5),  # Celsius
            'o2_saturation': 92  # minimum
        }

    def parse_medical_case(self, case_text):
        """Parse structured medical case data"""
        sections = {}
        current_section = None

        for line in case_text.split('\n'):
            if line.startswith('===='):
                section_name = line.strip('=').strip()
                current_section = section_name
                sections[current_section] = []
            elif current_section and line.strip():
                sections[current_section].append(line.strip())

        # Parse Q&A pairs
        qa_pairs = []
        if 'Question Answer Pair' in sections:
            qa_text = '\n'.join(sections['Question Answer Pair'])
            pairs = re.findall(r'Question \d+: (.*?)\nAnswer \d+: (.*?)(?=\n\d+|\Z)',
                               qa_text, re.DOTALL)
            qa_pairs = [{'question': q.strip(), 'answer': a.strip()}
                        for q, a in pairs]

        return {
            'summary': '\n'.join(sections.get('Summary of Current ER Visit', [])),
            'chief_complaint': '\n'.join(sections.get('Patient Chief Complaint', [])),
            'history': '\n'.join(sections.get('Patient History Summary', [])),
            'qa_pairs': qa_pairs,
            'visit_info': {
                line.split(': ')[0]: line.split(': ')[1]
                for line in sections.get('ER Visit Info', [])
                if ': ' in line
            }
        }

    def analyze_risk_factors(self, case_data):
        """Analyze risk factors from medical case data"""
        risk_score = 0
        risk_factors = []

        # Check chief complaint for high-risk conditions
        for condition in self.HIGH_RISK_CONDITIONS:
            if condition.lower() in case_data['chief_complaint'].lower():
                risk_score += 2
                risk_factors.append(f"High-risk condition: {condition}")

        # Analyze vital signs if available
        if 'visit_info' in case_data:
            try:
                acuity = float(case_data['visit_info'].get('Acuity', 5))
                risk_score += (5 - acuity) * 2  # Higher acuity = higher risk

                if case_data['visit_info'].get('Arrival Transport') == 'AMBULANCE':
                    risk_score += 1
                    risk_factors.append("Ambulance arrival")
            except ValueError:
                pass

        # Analyze Q&A responses for risk indicators
        for qa in case_data['qa_pairs']:
            if any(cond.lower() in qa['answer'].lower()
                   for cond in self.HIGH_RISK_CONDITIONS):
                risk_score += 1
                risk_factors.append(f"Risk indicator in Q&A: {qa['question']}")

        return risk_score, risk_factors

    def determine_esi_level(self, case_data):
        """Determine ESI level based on case analysis"""
        risk_score, risk_factors = self.analyze_risk_factors(case_data)

        # Generate ESI classification prompt
        prompt = PromptTemplate(
            template="""Based on the following patient information, determine the appropriate ESI level (1-5):

            Chief Complaint: {chief_complaint}

            Risk Score: {risk_score}
            Risk Factors: {risk_factors}

            Medical History: {history}

            Q&A Summary: {qa_summary}

            Guidelines:
            ESI 1: Immediate life-saving intervention required
            ESI 2: High risk situation or confused/lethargic/disoriented
            ESI 3: Multiple resources needed, but vital signs stable
            ESI 4: One resource needed
            ESI 5: No resources needed

            Provide ESI level and explanation:""",
            input_variables=["chief_complaint", "risk_score", "risk_factors",
                             "history", "qa_summary"]
        )

        # Prepare Q&A summary
        qa_summary = "\n".join([f"Q: {qa['question']}\nA: {qa['answer']}"
                                for qa in case_data['qa_pairs'][:3]])

        # Get LLM prediction
        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(
            chief_complaint=case_data['chief_complaint'],
            risk_score=risk_score,
            risk_factors="\n".join(risk_factors),
            history=case_data['history'],
            qa_summary=qa_summary
        )

        # Extract ESI level and confidence
        try:
            esi_level = int(re.search(r'ESI (\d)', response).group(1))
            confidence = self.calculate_confidence(response, case_data)
        except:
            esi_level = 3  # Default to middle level if parsing fails
            confidence = 0.5

        return {
            'esi_level': esi_level,
            'confidence': confidence,
            'explanation': response,
            'risk_factors': risk_factors,
            'needs_handoff': confidence < 0.85 or esi_level <= 2
        }

    def calculate_confidence(self, response, case_data):
        """Calculate confidence score for ESI prediction"""
        # Base confidence on risk analysis and response clarity
        base_confidence = 0.7

        # Adjust based on risk factors
        risk_score, _ = self.analyze_risk_factors(case_data)
        if risk_score > 5:
            base_confidence *= 0.8  # Reduce confidence for high-risk cases

        # Adjust based on data completeness
        if not case_data['qa_pairs']:
            base_confidence *= 0.9

        # Adjust based on response clarity
        if len(response) < 50 or "uncertain" in response.lower():
            base_confidence *= 0.9

        return min(base_confidence, 1.0)

    def generate_recommendations(self, esi_result, case_data):
        """Generate medical recommendations based on ESI assessment"""
        if esi_result['needs_handoff']:
            prompt = PromptTemplate(
                template="""Based on ESI level {esi_level} and the following factors:

                Risk Factors: {risk_factors}
                Chief Complaint: {chief_complaint}

                Generate:
                1. Immediate actions for medical staff
                2. Recommended labs/tests
                3. Patient monitoring requirements""",
                input_variables=["esi_level", "risk_factors", "chief_complaint"]
            )

            chain = LLMChain(llm=self.llm, prompt=prompt)
            recommendations = chain.run(
                esi_level=esi_result['esi_level'],
                risk_factors="\n".join(esi_result['risk_factors']),
                chief_complaint=case_data['chief_complaint']
            )

            return {
                'staff_recommendations': recommendations,
                'estimated_wait_time': self._calculate_wait_time(esi_result['esi_level']),
                'handoff_required': True
            }
        else:
            return {
                'staff_recommendations': "Standard protocol appropriate",
                'estimated_wait_time': self._calculate_wait_time(esi_result['esi_level']),
                'handoff_required': False
            }

    def _calculate_wait_time(self, esi_level):
        """Calculate estimated wait time based on ESI level"""
        base_times = {
            1: 0,  # Immediate
            2: 0.25,  # 15 minutes
            3: 1,  # 1 hour
            4: 2,  # 2 hours
            5: 3  # 3 hours
        }
        return base_times.get(esi_level, 2)  # Default to 2 hours if unknown


# # Example usage
# triage = MedicalTriageSystem()
#
# # Example case processing
# case_text = """[Your medical case text here]"""
# case_data = triage.parse_medical_case(case_text)
# esi_result = triage.determine_esi_level(case_data)
# recommendations = triage.generate_recommendations(esi_result, case_data)
#
# print(f"ESI Level: {esi_result['esi_level']}")
# print(f"Confidence: {esi_result['confidence']}")
# print(f"Handoff Required: {recommendations['handoff_required']}")
# print(f"Estimated Wait Time: {recommendations['estimated_wait_time']} hours")