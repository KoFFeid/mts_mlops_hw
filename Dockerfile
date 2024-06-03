# Select base image for the project
FROM python:3.12


WORKDIR /app

COPY requirements.txt ./

COPY /app /app
# Install all the requirements
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 3000

CMD uvicorn main:app --host=0.0.0.0 --port=3000 --reload  