FROM python:3.9

RUN pip install --upgrade pip

COPY . /app

WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install --no-cache-dir -r  requirements.txt 


ENV PORT 8080

CMD streamlit run app.py --server.port=${PORT}  --browser.serverAddress="0.0.0.0"

