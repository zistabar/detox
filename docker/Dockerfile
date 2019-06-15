FROM python:3
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app-docker
WORKDIR /app-docker
EXPOSE 5000
ENTRYPOINT ["python"]
CMD ["app-docker.py"]