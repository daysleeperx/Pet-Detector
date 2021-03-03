FROM python:3.7-slim-buster

WORKDIR /app

COPY get_pi_requirements.sh get_pi_requirements.sh
COPY requirements.txt requirements.txt
RUN chmod +x get_pi_requirements.sh
RUN ./get_pi_requirements.sh

COPY . .

CMD [ "python3", "detector.py"]
