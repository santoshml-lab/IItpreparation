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

# ================= INPUT =================
class SolveRequest(BaseModel):
    subject: str   # physics / math / chemistry
    question: str

# ================= CORE SOLVER =================
@app.post("/solve")
def solve(data: SolveRequest):

    prompt = f"""
You are an IIT-level expert teacher.

MISSION:
Solve problems with deep understanding.

RULES:
- First explain concept
- Then give approach
- Then step-by-step solution
- Then final answer
- No skipping steps
- Very logical IIT coaching style

FORMAT:

# 📘 Concept
# 🧠 Approach
# 🔢 Step-by-step Solution
# 🎯 Final Answer

Subject: {data.subject}

Question:
{data.question}
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": prompt}
        ],
        max_tokens=2000
    )

    return {
        "status": "success",
        "solution": res.choices[0].message.content
    }
    from pydantic import BaseModel

class PracticeRequest(BaseModel):
    subject: str
    question: str

@app.post("/practice")
def practice(data: PracticeRequest):

    prompt = f"""
You are IIT level Physics/Math/Chemistry teacher.

TASK:
1. Solve given problem step-by-step
2. Explain concept clearly
3. Then generate ONE similar but slightly tougher question

FORMAT STRICT:

# 📘 Solution
(complete step-by-step solution)

# 🧠 Concept Used
(explain concept briefly)

# 🔥 Next Practice Question
(generate similar IIT-level question ONLY)

Subject: {data.subject}
Question: {data.question}
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=2000
    )

    return {
        "status": "success",
        "result": res.choices[0].message.content
    }
    
