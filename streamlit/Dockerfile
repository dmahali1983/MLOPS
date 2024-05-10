# Base image
FROM python:3.10.12

# Set working directory
WORKDIR /app

# Copy files
COPY req_st.txt /app
COPY streamlit_app.py /app

copy . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r req_st.txt


# Run the application
EXPOSE 8501


#CMD waitress-serve --port 5000 --call "wsgi:create_app"
CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8501"]
