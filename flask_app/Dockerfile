# Base image
FROM python:3.10.12

# Set working directory
WORKDIR /app

# Copy files
COPY app.py /app
COPY req.txt /app
COPY predict.py /app


copy . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r req.txt
#RUN pip install gunicorn

ENV FLASK_APP = app.py

# Run the application
EXPOSE 5003

#CMD waitress-serve --port 5000 --call "wsgi:create_app"
#CMD ["sh", "-c", "cd /app && python3 app.py & streamlit run app_streamlit.py"]
#CMD ["gunicorn", "--bind", "0.0.0.0:5003", "app:app"]
CMD ["python", "app.py"]

