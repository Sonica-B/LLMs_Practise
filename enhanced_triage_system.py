from langchain_ollama import OllamaLLM
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
import re
import json
# Import original triage system
from LLMs_Practise.LLMs_Practise.TriageSystem import MedicalTriageSystem


class EnhancedMedicalTriageSystem(MedicalTriageSystem):
    def __init__(self):
        super().__init__()

        # Define diagnostic categories for possible diagnoses
        self.DIAGNOSTIC_CATEGORIES = {
            'cardiovascular': ['chest pain', 'shortness of breath', 'palpitations', 'hypertension'],
            'respiratory': ['cough', 'shortness of breath', 'pneumonia', 'asthma', 'copd'],
            'neurological': ['headache', 'dizziness', 'seizure', 'stroke', 'confusion', 'altered mental status'],
            'gastrointestinal': ['abdominal pain', 'vomiting', 'diarrhea', 'constipation', 'gi bleed'],
            'musculoskeletal': ['pain', 'fall', 'fracture', 'sprain', 'back pain'],
            'infectious': ['fever', 'infection', 'sepsis', 'cellulitis']
        }

        # Initialize handoff reason categories
        self.HANDOFF_CATEGORIES = [
            "Critical physiological instability",
            "Complex medical history requiring expert interpretation",
            "Diagnostic uncertainty with high risk",
            "Resource limitations for critical care",
            "Patient deterioration risk",
            "Special population considerations (pediatric, geriatric, pregnant)",
            "Protocol-mandated physician assessment"
        ]

    def determine_esi_level(self, case_data):
        """Enhanced ESI level determination with detailed handoff analysis"""
        result = super().determine_esi_level(case_data)

        # Add detailed handoff analysis
        result['handoff_analysis'] = self.analyze_handoff_requirements(result, case_data)

        # Add possible diagnoses
        result['possible_diagnoses'] = self.predict_possible_diagnoses(case_data, result)

        return result

    def analyze_handoff_requirements(self, esi_result, case_data):
        """Analyze and explain handoff requirements in detail"""
        if not esi_result['needs_handoff']:
            return {"required": False, "reasons": ["Patient stable and within AI system capabilities"]}

        prompt = PromptTemplate(
            template="""Based on the following patient information, explain WHY a handoff to a human medical provider is required:

            Chief Complaint: {chief_complaint}
            Summary: {summary}
            Risk Factors: {risk_factors}
            ESI Level: {esi_level}

            For each applicable reason below, provide a brief explanation of how it applies to this case:
            1. Critical physiological instability
            2. Complex medical history requiring expert interpretation
            3. Diagnostic uncertainty with high risk
            4. Resource limitations for critical care
            5. Patient deterioration risk
            6. Special population considerations (pediatric, geriatric, pregnant)
            7. Protocol-mandated physician assessment

            Format your response as a structured JSON with "primary_reason" and "reasons" as an array of objects with "category" and "explanation" fields.
            """,
            input_variables=["chief_complaint", "summary", "risk_factors", "esi_level"]
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(
            chief_complaint=case_data['chief_complaint'],
            summary=case_data['summary'][:400],
            risk_factors="\n".join(esi_result['risk_factors']),
            esi_level=esi_result['esi_level']
        )

        try:
            # Extract JSON from response
            json_str = re.search(r'({.*})', response.replace('\n', ' '), re.DOTALL)
            if json_str:
                handoff_data = json.loads(json_str.group(1))
                return {
                    "required": True,
                    "primary_reason": handoff_data.get("primary_reason", "High risk situation"),
                    "reasons": handoff_data.get("reasons", [{"category": "High risk situation",
                                                             "explanation": "ESI level and risk factors indicate handoff is required"}])
                }
        except:
            pass

        # Fallback if JSON parsing fails
        return {
            "required": True,
            "primary_reason": "High risk patient (ESI level " + str(esi_result['esi_level']) + ")",
            "reasons": [{"category": c, "explanation": "May apply based on risk factors"} for c in
                        self.HANDOFF_CATEGORIES[:3]]
        }

    def predict_possible_diagnoses(self, case_data, esi_result):
        """Predict possible diagnoses based on symptoms and risk factors"""
        combined_text = f"{case_data['chief_complaint']} {case_data['summary']}"

        prompt = PromptTemplate(
            template="""Based on the following information, list the top 3-5 most likely diagnoses:

            Chief Complaint: {chief_complaint}
            Summary: {summary}
            Risk Factors: {risk_factors}
            ESI Level: {esi_level}

            Format your response as a JSON array of objects with "diagnosis" and "likelihood" (high, medium, low) fields.
            """,
            input_variables=["chief_complaint", "summary", "risk_factors", "esi_level"]
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(
            chief_complaint=case_data['chief_complaint'],
            summary=case_data['summary'][:400],
            risk_factors="\n".join(esi_result['risk_factors']),
            esi_level=esi_result['esi_level']
        )

        try:
            # Extract JSON from response
            json_str = re.search(r'(\[.*\])', response.replace('\n', ' '), re.DOTALL)
            if json_str:
                return json.loads(json_str.group(1))
        except:
            pass

        # Fallback if JSON parsing fails - use rule-based approach
        diagnoses = []
        for category, keywords in self.DIAGNOSTIC_CATEGORIES.items():
            for keyword in keywords:
                if keyword.lower() in combined_text.lower():
                    diagnoses.append({
                        "diagnosis": f"Possible {category} issue",
                        "likelihood": "medium"
                    })
                    break

        return diagnoses[:5]  # Return top 5

    def generate_recommendations(self, esi_result, case_data):
        """Generate enhanced medical recommendations with structured fields"""
        base_recs = super().generate_recommendations(esi_result, case_data)

        # Generate structured recommendations
        prompt = PromptTemplate(
            template="""Based on:
            ESI Level: {esi_level}
            Risk Factors: {risk_factors}
            Chief Complaint: {chief_complaint}
            Summary: {summary}
            Possible Diagnoses: {diagnoses}

            Generate the following as JSON:
            1. "labs": Array of recommended laboratory tests with rationale
            2. "imaging": Array of recommended imaging studies with rationale
            3. "medications": Array of potential medications to consider
            4. "monitoring": Recommended vital signs monitoring frequency and parameters
            5. "consults": Any specialist consultations that might be needed
            """,
            input_variables=["esi_level", "risk_factors", "chief_complaint", "summary", "diagnoses"]
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(
            esi_level=esi_result['esi_level'],
            risk_factors="\n".join(esi_result['risk_factors']),
            chief_complaint=case_data['chief_complaint'],
            summary=case_data['summary'][:400],
            diagnoses=json.dumps(esi_result.get('possible_diagnoses', []))
        )

        structured_recs = {}
        try:
            # Extract JSON from response
            json_str = re.search(r'({.*})', response.replace('\n', ' '), re.DOTALL)
            if json_str:
                structured_recs = json.loads(json_str.group(1))
        except:
            # Fallback with basic structured recommendations
            structured_recs = {
                "labs": [{"name": "Basic metabolic panel", "rationale": "Baseline assessment"}],
                "imaging": [],
                "medications": [],
                "monitoring": "Standard vital signs",
                "consults": []
            }

        # Combine with base recommendations
        return {
            **base_recs,
            "structured_recommendations": structured_recs,
            "handoff_required": esi_result['needs_handoff'],
            "handoff_analysis": esi_result.get('handoff_analysis', {})
        }