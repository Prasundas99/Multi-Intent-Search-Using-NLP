from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Body, Depends

from datetime import datetime
import uvicorn

from DTO.searchRequest import SearchRequest

deployedTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "deployedTime": deployedTime,
        "message": "Hello World"
    }


@app.post("/search")
async def search(request: SearchRequest):
    query = request.text
    return {"query": query, "message": "Search functionality not implemented yet"}


# Run the API with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)







