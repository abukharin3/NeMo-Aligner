import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import uvicorn
import argparse
import time

# pip install fastapi; pip install -U transformers; pip install uvicorn

# Load model and tokenizer
model_name = "/lustre/fsw/portfolios/llmservice/users/abukharin/reward_hacking/Skywork/Skywork-Reward-Gemma-2-27B-v0.2"
rm = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    num_labels=1,
)
rm_tokenizer = AutoTokenizer.from_pretrained(model_name)

# API definition
app = FastAPI()

class Query(BaseModel):
    conversation: list

@app.post("/get_reward")
async def get_reward(query: Query):
    # Format conversation
    conversation = query.conversation
    #conversation = [{"role": "user", "content": query.prompt}, {"role": "assistant", "content": query.response}]
    formatted = rm_tokenizer.apply_chat_template(conversation, tokenize=False)
    tokenized = rm_tokenizer(formatted, return_tensors="pt").to(device)

    # Compute reward
    with torch.no_grad():
        score = rm(**tokenized).logits[0][0].item()

    return {"reward": score}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the reward model server.")
    parser.add_argument("--port", type=int, default=5003, help="Port number to run the server on.")
    args = parser.parse_args()

    print("Starting reward model server...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)

    
    time.sleep(36000)