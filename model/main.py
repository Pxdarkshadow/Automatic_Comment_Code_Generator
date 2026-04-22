# model/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import base64
import json

app = FastAPI()

class CodeRequest(BaseModel):
    code: str

@app.post("/predict")
def predict_comment(request: CodeRequest):
    try:
        code_b64 = base64.b64encode(request.code.encode('utf-8')).decode('utf-8')
        
        command = [
            "python3", "predict.py", "--b64", code_b64, "--json", 
            "--mode", "beam", "--beam-width", "6", "--temperature", "0.65", 
            "--min-len", "8", "--max-len", "48", "--length-alpha", "0.7", "--repetition-penalty", "1.3"
        ]
        
        # Note: In the AWS environment, 'predict.py' should be in the same folder as this execution.
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        # Parse the JSON response
        return json.loads(result.stdout.strip())
        
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e.stderr}")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail="Failed to parse model output as JSON")