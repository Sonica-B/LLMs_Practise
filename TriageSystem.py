import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import numpy as np


class TriageSystem:
    def __init__(self):
        # Initialize Ollama LLM
        self.llm = Ollama(model="mistral")

        # Initialize vector store for RAG
        self.embeddings = HuggingFaceEmbeddings()
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory="./triage_db"
        )

        # Define confidence thresholds
        self.HANDOFF_THRESHOLD = 0.85
        self.MIN_QUESTIONS = 3

    def calculate_confidence(self, response, context):
        """Calculate confidence score using difficulty-assisted encoding"""
        # Analyze response complexity
        complexity_score = self._assess_complexity(response)

        # Calculate semantic similarity with context
        similarity_score = self._calculate_similarity(response, context)

        # Weighted confidence score
        confidence = (0.7 * similarity_score + 0.3 * (1 - complexity_score))
        return confidence

    def _assess_complexity(self, text):
        """Assess text complexity using metrics similar to DAMI"""
        # Implementation of difficulty-assisted encoding
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / len(words)
        unique_words = len(set(words))

        complexity = (avg_word_length * 0.4 + (unique_words / len(words)) * 0.6)
        return min(complexity, 1.0)

    def classify_esi(self, patient_data):
        """Classify ESI level with confidence estimation"""
        # Retrieve relevant cases from vector store
        similar_cases = self.vectorstore.similarity_search(
            patient_data,
            k=5
        )

        # Generate ESI classification prompt
        prompt = PromptTemplate(
            template="""Based on the following patient data and similar cases,
            determine the ESI level (1-5):
            Patient Data: {patient_data}
            Similar Cases: {similar_cases}
            Provide ESI level and confidence score.""",
            input_variables=["patient_data", "similar_cases"]
        )

        # Get LLM prediction
        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(
            patient_data=patient_data,
            similar_cases=similar_cases
        )

        # Calculate confidence
        confidence = self.calculate_confidence(response, patient_data)

        return {
            'esi_level': self._extract_esi(response),
            'confidence': confidence,
            'needs_handoff': confidence < self.HANDOFF_THRESHOLD
        }

    def generate_reports(self, classification_result, patient_data):
        """Generate dual reports for ER staff and patients"""
        if classification_result['needs_handoff']:
            # Generate handoff report for ER staff
            er_report = self._generate_er_report(
                classification_result,
                patient_data
            )

            # Generate limited patient questions
            patient_questions = self._generate_patient_questions(
                classification_result,
                patient_data
            )

            return {
                'er_report': er_report,
                'patient_questions': patient_questions,
                'handoff_recommended': True
            }
        else:
            # Generate standard reports
            return self._generate_standard_reports(
                classification_result,
                patient_data
            )

    def _generate_er_report(self, classification, patient_data):
        """Generate detailed report for ER staff"""
        prompt = PromptTemplate(
            template="""Generate a detailed ER report based on:
            ESI Level: {esi_level}
            Confidence: {confidence}
            Patient Data: {patient_data}
            Include key observations and recommendations.""",
            input_variables=["esi_level", "confidence", "patient_data"]
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(
            esi_level=classification['esi_level'],
            confidence=classification['confidence'],
            patient_data=patient_data
        )

    def _generate_patient_questions(self, classification, patient_data):
        """Generate follow-up questions for patient"""
        prompt = PromptTemplate(
            template="""Based on ESI level {esi_level} and:
            Patient Data: {patient_data}
            Generate 3-5 key follow-up questions to gather more information.""",
            input_variables=["esi_level", "patient_data"]
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(
            esi_level=classification['esi_level'],
            patient_data=patient_data
        )


# Usage example
triage = TriageSystem()

# Process patient
patient_data = """
Chief complaint: Chest pain
Vital signs: BP 140/90, HR 95, RR 18, T 37.2C
Additional symptoms: Shortness of breath, left arm pain
"""

# Classify and generate reports
classification = triage.classify_esi(patient_data)
reports = triage.generate_reports(classification, patient_data)

# Check if handoff needed
if classification['needs_handoff']:
    print("Handoff to human provider recommended")
    print(f"Confidence: {classification['confidence']}")