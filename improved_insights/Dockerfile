FROM python:3.8
LABEL maintainer "Olorunleke"
RUN pip install --upgrade pip
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY . /app
ENV PORT 8080
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
