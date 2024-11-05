from fastapi import FastAPI
from fastapi import Query

app = FastAPI()

@app.get("/")
async def root(input_text: str = Query(..., description="Input text from the user")):
    return {"echo": input_text}