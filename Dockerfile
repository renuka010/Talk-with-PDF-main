FROM python:3.10
WORKDIR /code
RUN apt update && apt upgrade -y
COPY ./Talk-with-PDF-main/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./Talk-with-PDF-main /code

CMD ["streamlit", "run", "app.py"]
