from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ================= INPUT MODELS =================

class SolveRequest(BaseModel):
    subject: str
    question: str


class PracticeRequest(BaseModel):
    subject: str
    question: str


class EvaluateRequest(BaseModel):
    subject: str
    question: str
    student_answer: str


# ================= GLOBAL PROMPT RULES =================

BASE_RULES = """
You are an IIT-level expert teacher.

STRICT RULES:
- Use ONLY given values from question
- Do NOT assume missing values
- No extra theory outside relevance
- Keep answer clean and structured
- Use plain text only (no symbols like μ, √, ²)
- Use mu, sqrt, ^2 instead
- Step-by-step reasoning required
- IIT coaching style explanation
- Output must be COPY-FRIENDLY (no markdown widgets)
"""


# ================= SOLVE =================

@app.post("/solve")
def solve(data: SolveRequest):

    prompt = f"""
{BASE_RULES}

MISSION:
Solve the problem in structured IIT format.

FORMAT:
Concept
Approach
Step-by-step Solution
Final Answer

Subject: {data.subject}

Question:
{data.question}
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=2000
    )

    return {
        "status": "success",
        "solution": res.choices[0].message.content.strip()
    }


# ================= PRACTICE =================

@app.post("/practice")
def practice(data: PracticeRequest):

    prompt = f"""
{BASE_RULES}

TASK:
1. Solve step-by-step
2. Explain concept simply
3. Generate ONE similar tougher IIT question

FORMAT:
Solution
Concept Used
Next Practice Question

Subject: {data.subject}

Question:
{data.question}
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=2000
    )

    return {
        "status": "success",
        "result": res.choices[0].message.content.strip()
    }


# ================= EVALUATE =================

@app.post("/evaluate")
def evaluate(data: EvaluateRequest):

    prompt = f"""
{BASE_RULES}

TASK:
Evaluate student's answer like IIT mentor.

FORMAT:
Answer Evaluation
Correct Parts
Mistakes
Correct Concept
Score out of 10
Improvement Tip

Subject: {data.subject}

Question:
{data.question}

Student Answer:
{data.student_answer}
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=2000
    )

    return {
        "status": "success",
        "result": res.choices[0].message.content.strip()
    }
