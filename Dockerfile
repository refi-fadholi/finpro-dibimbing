FROM python:3.8 AS python-base
ENV ENV=production \
    PORT=8001
RUN python -m pip install --upgrade pip
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /app
COPY . /app
EXPOSE 8001
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
