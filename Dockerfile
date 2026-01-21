FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app
COPY templates/ templates/


RUN pip install -r requirements.txt

CMD ["python3", "app.py"]