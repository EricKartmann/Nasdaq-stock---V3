FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
COPY nasdaq_stocks.py .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "nasdaq_stocks.py", "--server.port=8501", "--server.address=0.0.0.0"] 