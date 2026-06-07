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

# ================= MODELS =================

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


class SummaryRequest(BaseModel):
    subject: str
    topic: str


# ================= BASE RULES =================

BASE_RULES = """
You are an IIT-level expert teacher.

STRICT RULES:

- Use only given values
- No assumptions
- Step-by-step reasoning required
- Clean structured IIT coaching style
- Keep answers exam focused

VERIFICATION RULES:

- Verify every calculation before final answer
- Check algebraic manipulations carefully
- Check signs (+/-) carefully
- Check units and dimensions where applicable
- Recalculate final numerical answer once again
- Never guess values
- Never skip logical steps
- If multiple methods exist, use the most reliable one

FOR MATHEMATICS:

- Verify identities before using them
- Perform reverse checking whenever possible
- Recheck substitutions and simplifications

FOR PHYSICS:

- Check units
- Check dimensions
- Check direction/sign conventions
- Verify formulas before substitution

FOR CHEMISTRY:

- Verify balancing
- Verify mole calculations
- Verify unit conversions
- Verify final numerical values
CONTRADICTION CHECK:

- If two different values are obtained for the same quantity, stop and re-evaluate
- Never continue with conflicting results
- Resolve contradictions before giving final answer
- If an intermediate value changes, recompute all dependent steps

FINAL WORKFLOW:

1. Understand problem
2. Identify concept
3. Solve step by step
4. Verify each intermediate result
5. Check for contradictions
6. Recalculate final answer independently
7. Compare both answers
8. If answers match, provide final answer
9. If answers differ, solve again
"""













# ================= SOLVE =================

@app.post("/solve")
def solve(data: SolveRequest):

    prompt = f"""
{BASE_RULES}

MISSION:
Solve step-by-step.

FORMAT:
# Concept
# Approach
# Step-by-step Solution
# Verification
# Final Answer


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
3. Give ONE tougher similar question

FORMAT:
# Solution
# Concept Used
# Next Practice Question

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
Evaluate student answer.

FORMAT:
# Answer Evaluation
# Correct Parts
# Mistakes
# Correct Concept
# Score (out of 10)
# Improvement Tip

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


# ================= HINT =================

@app.post("/hint")
def hint(data: SolveRequest):

    prompt = f"""
You are an IIT mentor.

TASK:
Give guided hints only (NO full solution).

FORMAT:
# Hint 1
# Hint 2
# Hint 3
# Final Direction

Subject: {data.subject}

Question:
{data.question}
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=1200
    )

    return {
        "status": "success",
        "result": res.choices[0].message.content.strip()
    }


# ================= SUMMARY =================

@app.post("/summary")
def summary(data: SummaryRequest):

    prompt = f"""
You are an IIT revision coach.

TASK:
Create short exam-ready revision notes.

FORMAT:
# Summary Notes
# Key Points
# Important Formulas / Concepts
# Exam Tips

Subject: {data.subject}

Topic:
{data.topic}
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=1200
    )

    return {
        "status": "success",
        "result": res.choices[0].message.content.strip()
    }
    # ================= TOPIC LEARN MODELS =================

class TopicLearnRequest(BaseModel):
    subject: str
    topic: str


class MCQCheckRequest(BaseModel):
    selected_answer: str
    correct_answer: str


# ================= TOPIC LEARN =================

@app.post("/topic-learn")
def topic_learn(data: TopicLearnRequest):

    prompt = f"""
You are an IIT-level teacher.

Teach the topic deeply.

FORMAT:

# Concept

# Important Formulae

# Common Mistakes

MCQ RULES:

- Generate ONLY conceptual theory MCQs.
- Do NOT ask derivative calculations.
- Do NOT ask integration calculations.
- Do NOT ask numerical problems.
- Do NOT ask formula evaluation.
- Do NOT ask multi-step mathematics.
- Questions should test concepts only.
- Answer should be identifiable from theory.

Question:
...

Options:
A)
B)
C)
D)

Correct Answer:
...

Subject:
{data.subject}

Topic:
{data.topic}
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": prompt
            }
        ],
        max_tokens=2000
    )

    return {
        "status": "success",
        "result": res.choices[0].message.content.strip()
    }


# ================= MCQ CHECK =================

@app.post("/check-mcq")
def check_mcq(data: MCQCheckRequest):

    correct = (
        data.selected_answer.strip().upper()
        ==
        data.correct_answer.strip().upper()
    )

    return {
        "correct": correct,
        "message": "Correct ✅" if correct else "Incorrect ❌"
    }
   


    
    






    
    
    
    





    
    
    


    


    




