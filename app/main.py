from fastapi import FastAPI

app = FastAPI(title="Biz-AI")


@app.get("/health")
def health():
    return {"ok": True}
