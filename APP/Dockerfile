FROM python:3.8.3
WORKDIR /app
ADD . /app
RUN pip --no-cache-dir install -r /app/requirements.txt
EXPOSE 5000
ENV NAME OpentoAll
CMD ["python","app.py"]
