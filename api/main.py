from fastapi import FastAPI
from api.routes import segment, track, follow

app = FastAPI()
app.include_router(segment.router)
app.include_router(track.router)
app.include_router(follow.router)

# helper
@app.get("/")
def root():
    return {"status": "ok"}
