# Customer Support Agent (CrewAI)

A beginner-friendly customer support assistant built with CrewAI. Three AI agents work together:

1. **Support Agent** answers from a company FAQ file (simple RAG).
2. **Research Agent** searches the web when the FAQ has no answer. Questions about the company itself are passed to the support team instead of guessed.
3. **Record Keeper** saves every question, answer and source in a Google Sheet.

`main.py` is a live chat in the terminal. `main_batch.py` answers several questions at the same time (parallel).

## Requirements

- Python 3.10 to 3.13
- OpenAI API key
- Serper API key (serper.dev) for web search
- A Google Sheet and a Google service account (credentials.json)

## Setup (Windows)

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

1. Copy `.env.example` to `.env` and fill in your keys.
2. Put your Google service account file in the project folder as `credentials.json`.
3. Create a Google Sheet with the headings: Date Time, Question, Answer, Source.
4. Share the sheet with the `client_email` from `credentials.json` as Editor.
5. Edit `knowledge/faq.txt` with your own company FAQs.

## Run

```
python main.py
```

Type a question. Type `exit` to stop.

## Security

Never upload `.env` or `credentials.json`. They are listed in `.gitignore`.