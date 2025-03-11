from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

class Agent:
    def __init__(self, medical_report=None, role=None, extra_info=None):
        self.medical_report = medical_report
        self.role = role
        self.extra_info = extra_info
        # Initialize the prompt based on role and extra info
        self.prompt_template = self.create_prompt_template()
        # Initialize the model (using GPT-4 OpenAI)
        self.model = ChatOpenAI(temperature=0, model="gpt-4o")

    def create_prompt_template(self):
        if self.role == "MultidisciplinaryTeam":
            template = f"""
Act as a multidisciplinary team of healthcare professionals.
You will receive medical reports from the following specialties:
- Cardiologist
- Psychologist
- Pulmonologist
- Dermatologist
- Neurologist
- Gastroenterologist
- Endocrinologist
- Orthopedist
- Nephrologist
- Oncologist

Task: Review these reports, analyze them comprehensively, and provide a bullet-point list of potential health issues. For each issue, indicate:
  - The specialty (or specialties) that raised the concern.
  - A brief explanation of the findings.
  - Recommended diagnostic follow-ups.
  - Suggested treatment options (such as medications, lifestyle modifications, surgical interventions, or other therapies) when applicable.

Cardiologist Report: {self.extra_info.get('cardiologist_report', '')}
Psychologist Report: {self.extra_info.get('psychologist_report', '')}
Pulmonologist Report: {self.extra_info.get('pulmonologist_report', '')}
Dermatologist Report: {self.extra_info.get('dermatologist_report', '')}
Neurologist Report: {self.extra_info.get('neurologist_report', '')}
Gastroenterologist Report: {self.extra_info.get('gastroenterologist_report', '')}
Endocrinologist Report: {self.extra_info.get('endocrinologist_report', '')}
Orthopedist Report: {self.extra_info.get('orthopedist_report', '')}
Nephrologist Report: {self.extra_info.get('nephrologist_report', '')}
Oncologist Report: {self.extra_info.get('oncologist_report', '')}
            """
        else:
            templates = {
                "Cardiologist": """
Act as a cardiologist. You will receive a patient's medical report.
Task: Review the cardiac workup (ECG, blood tests, echocardiogram, etc.), identify potential cardiac issues (such as arrhythmias, structural abnormalities, or ischemic changes), and recommend further diagnostic tests and treatment options (medications, lifestyle modifications, or interventional procedures) where applicable.
Medical Report: {medical_report}
                """,
                "Psychologist": """
Act as a psychologist. You will receive a patient's medical report.
Task: Analyze the report to identify potential mental health issues (such as anxiety, depression, trauma, etc.) and recommend appropriate interventions, including counseling, psychotherapy, or pharmacotherapy.
Patient's Report: {medical_report}
                """,
                "Pulmonologist": """
Act as a pulmonologist. You will receive a patient's medical report.
Task: Evaluate the report for respiratory issues such as asthma, COPD, lung infections, or other conditions, and suggest further diagnostic steps and treatment options (inhalers, medications, pulmonary rehabilitation, etc.) as needed.
Patient's Report: {medical_report}
                """,
                "Dermatologist": """
Act as a dermatologist. You will receive a patient's medical report.
Task: Review the report for skin conditions such as eczema, psoriasis, acne, or other dermatological issues, and recommend further evaluation and treatment options (topical therapies, systemic medications, etc.).
Patient's Report: {medical_report}
                """,
                "Neurologist": """
Act as a neurologist. You will receive a patient's medical report.
Task: Analyze the report for neurological issues such as migraines, seizures, neuropathy, or other conditions, and recommend further evaluations and treatment options (medications, lifestyle adjustments, or other therapies) as appropriate.
Patient's Report: {medical_report}
                """,
                "Gastroenterologist": """
Act as a gastroenterologist. You will receive a patient's medical report.
Task: Evaluate the report for gastrointestinal issues such as GERD, IBS, gastritis, or other conditions, and suggest further diagnostic tests and treatment options (dietary changes, medications, endoscopic procedures, etc.) when needed.
Patient's Report: {medical_report}
                """,
                "Endocrinologist": """
Act as an endocrinologist. You will receive a patient's medical report.
Task: Review the report for endocrine and metabolic issues such as diabetes, thyroid disorders, or hormonal imbalances, and recommend further testing and management options (medications, lifestyle changes, or other therapies) as required.
Patient's Report: {medical_report}
                """,
                "Orthopedist": """
Act as an orthopedist. You will receive a patient's medical report.
Task: Analyze the report for musculoskeletal issues such as joint pain, fractures, arthritis, or other orthopedic conditions, and suggest further evaluation and treatment options (physiotherapy, medications, surgical interventions, etc.) when appropriate.
Patient's Report: {medical_report}
                """,
                "Nephrologist": """
Act as a nephrologist. You will receive a patient's medical report.
Task: Evaluate the report for kidney-related issues such as chronic kidney disease, electrolyte imbalances, or other renal conditions, and recommend further diagnostic tests and treatment options (medications, dietary modifications, or other therapies) as needed.
Patient's Report: {medical_report}
                """,
                "Oncologist": """
Act as an oncologist. You will receive a patient's medical report.
Task: Review the report for any indications of cancer or precancerous conditions. Diagnose accurately and, if a malignancy is detected or suspected, suggest appropriate diagnostic follow-ups and treatment options, including surgery, chemotherapy, radiotherapy, targeted therapies, or immunotherapy.
Patient's Report: {medical_report}
                """
            }
            template = templates[self.role]
        return PromptTemplate.from_template(template)
    
    def run(self):
        print(f"{self.role} is running...")
        prompt = self.prompt_template.format(medical_report=self.medical_report)
        try:
            response = self.model.invoke(prompt)
            return response.content
        except Exception as e:
            print("Error occurred:", e)
            return None

# Specialized agent classes for each medical specialty
class Cardiologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Cardiologist")

class Psychologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Psychologist")

class Pulmonologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Pulmonologist")

class Dermatologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Dermatologist")

class Neurologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Neurologist")

class Gastroenterologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Gastroenterologist")

class Endocrinologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Endocrinologist")

class Orthopedist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Orthopedist")

class Nephrologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Nephrologist")

class Oncologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Oncologist")

class MultidisciplinaryTeam(Agent):
    def __init__(self, cardiologist_report, psychologist_report, pulmonologist_report, dermatologist_report, neurologist_report, gastroenterologist_report, endocrinologist_report, orthopedist_report, nephrologist_report, oncologist_report):
        extra_info = {
            "cardiologist_report": cardiologist_report,
            "psychologist_report": psychologist_report,
            "pulmonologist_report": pulmonologist_report,
            "dermatologist_report": dermatologist_report,
            "neurologist_report": neurologist_report,
            "gastroenterologist_report": gastroenterologist_report,
            "endocrinologist_report": endocrinologist_report,
            "orthopedist_report": orthopedist_report,
            "nephrologist_report": nephrologist_report,
            "oncologist_report": oncologist_report
        }
        super().__init__(role="MultidisciplinaryTeam", extra_info=extra_info)
