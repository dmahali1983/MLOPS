# Base image

FROM python:3.10-slim
#From pytorch/pytorch:1.9.0-cpu

WORKDIR /app
# Copy files
COPY . /app

RUN  python -m pip install --upgrade pip
RUN  pip3 install -r req.txt

ENV FLASK_APP = app.py

# Run the application
EXPOSE 8080

#CMD ["python", "app.py"]
ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]


