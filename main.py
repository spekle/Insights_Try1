from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (you can restrict this to specific domains)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def root(input_text: str = Query(..., description="Input text from the user")):
    return {"echo": input_text}
