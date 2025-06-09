# Voxarch

Multimodal QA system over books and audio.  
Ask questions and get answers with cited sources (text/audio).

- Python backend (FastAPI), React frontend.
- Place files in `data/text/` and `data/audio/`.
- Build index: `python -m voxarch.scripts.build_index`
- Run: `uvicorn voxarch.api.main:app --reload` and `npm start` in frontend.

