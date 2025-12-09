from langchain_ollama import OllamaLLM
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
import re
import json
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
        # Get base ESI result from parent method
        result = super().determine_esi_level(case_data)

        # Replace simple handoff logic with enhanced decision system
        handoff_analysis = self.determine_handoff_requirement(result, case_data)
        result['handoff_analysis'] = handoff_analysis

        # Update the needs_handoff flag based on the detailed analysis
        result['needs_handoff'] = handoff_analysis['required']

        # Add possible diagnoses
        result['possible_diagnoses'] = self.predict_possible_diagnoses(case_data, result)

        return result

    def determine_handoff_requirement(self, esi_result, case_data):
        """
        Determine if a handoff to a human provider is required based on multiple factors
        """
        # Extract key decision factors
        esi_level = esi_result['esi_level']
        confidence = esi_result['confidence']
        risk_factors = esi_result.get('risk_factors', [])
        risk_score = esi_result.get('risk_score', 0)

        # 1. Critical cases always require handoff (ESI 1-2)
        if esi_level <= 2:
            return {
                "required": True,
                "primary_reason": f"High acuity case (ESI level {esi_level})",
                "confidence_score": 0.95,
                "reasons": [
                    {"category": "Critical physiological instability",
                     "explanation": "ESI levels 1-2 indicate potentially life-threatening conditions requiring immediate physician evaluation"}
                ]
            }

        # 2. Check for critical risk factors that would necessitate handoff
        critical_keywords = ["altered mental status", "chest pain", "shortness of breath",
                             "severe pain", "confusion", "encephalopathy"]
        critical_risks = [factor for factor in risk_factors
                          if any(keyword in factor.lower() for keyword in critical_keywords)]

        if critical_risks and esi_level == 3:
            return {
                "required": True,
                "primary_reason": "Critical risk factors present",
                "confidence_score": 0.90,
                "reasons": [
                    {"category": "Patient deterioration risk",
                     "explanation": f"Critical symptoms detected: {', '.join(critical_risks)}"}
                ]
            }

        # 3. Confidence-based decision making with escalating thresholds by ESI level
        confidence_thresholds = {
            3: 0.70,  # ESI 3 requires higher confidence to avoid handoff
            4: 0.65,  # ESI 4 can be handled with moderate confidence
            5: 0.60  # ESI 5 can be handled with lower confidence
        }

        if confidence < confidence_thresholds.get(esi_level, 0.75):
            return {
                "required": True,
                "primary_reason": "Low confidence in AI assessment",
                "confidence_score": 0.80,
                "reasons": [
                    {"category": "Diagnostic uncertainty with high risk",
                     "explanation": f"AI confidence ({confidence:.2f}) below acceptable threshold for ESI {esi_level}"}
                ]
            }

        # 4. Special case handling - arrival method
        arrival_method = case_data.get('visit_info', {}).get('Arrival Transport', '').upper()
        if arrival_method == 'AMBULANCE' and esi_level <= 3:
            return {
                "required": True,
                "primary_reason": "Protocol-mandated assessment for ambulance arrivals",
                "confidence_score": 0.85,
                "reasons": [
                    {"category": "Protocol-mandated physician assessment",
                     "explanation": "Ambulance arrival with ESI ≤ 3 requires physician evaluation per protocol"}
                ]
            }

        # 5. Lower ESI level (4-5) with good confidence and no critical factors - no handoff required
        if esi_level >= 4 and confidence >= 0.75 and not critical_risks:
            return {
                "required": False,
                "primary_reason": "Stable non-urgent case within AI capabilities",
                "confidence_score": 0.85,
                "reasons": [
                    "Low acuity case (ESI 4-5) with high AI confidence",
                    "No critical risk factors identified",
                    "Case suitable for algorithmic management with standard protocols"
                ]
            }

        # Default conservative approach for edge cases
        return {
            "required": esi_level <= 3,
            "primary_reason": f"ESI level {esi_level} case with standard protocols",
            "confidence_score": 0.75,
            "reasons": [
                {"category": "Standard protocol application",
                 "explanation": f"Following standard handoff guidelines for ESI level {esi_level} cases"}
            ]
        }

    def analyze_handoff_requirements(self, esi_result, case_data):
        """Analyze and explain handoff requirements in detail - deprecated, use determine_handoff_requirement instead"""
        # This method is kept for backward compatibility
        handoff_analysis = self.determine_handoff_requirement(esi_result, case_data)
        return handoff_analysis

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

        # Get handoff requirement information
        handoff_required = esi_result['needs_handoff']
        handoff_analysis = esi_result.get('handoff_analysis', {})

        # Combine with base recommendations
        recommendations = {
            **base_recs,
            "structured_recommendations": structured_recs,
            "handoff_required": handoff_required,
            "handoff_analysis": handoff_analysis
        }

        # Adjust wait time based on handoff decision for a more nuanced approach
        if handoff_required and recommendations['estimated_wait_time'] > 0.5:
            # Reduce wait time for handoff cases to ensure they're seen sooner
            recommendations['estimated_wait_time'] *= 0.8

        return recommendations