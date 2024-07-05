from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

from datetime import datetime
from nlpModels.lstm import nlpSearch, trainLstm
from nlpModels.word2vec import train_word2vec
import uvicorn

from DTO.searchRequest import SearchRequest
from data.corpus import generateCorpus, corpus

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
    generateCorpus()
    return {
        "deployedTime": deployedTime,
        "message": "Hello World"
    }


@app.post("/search")
async def search(request: SearchRequest):
    print("Request received: " + request.text)

    print("Generating corpus")
    corpusData = generateCorpus()
    print("Corpus generated")

    print("Training word2vec model")
    w2vModel = train_word2vec(corpusData)
    print("Word2vec model trained")

    print("Training LSTM")
    model, tokenizer, max_sequence_length = trainLstm(corpusData)
    print("LSTM model trained")

    print("All models trained")
    results = nlpSearch(request.text, model,w2vModel, tokenizer, max_sequence_length)

    print(f"Results for '{query}':")
    print(results)

    query = request.text
    return {"query": query, "results": results}


# Run the API with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)







