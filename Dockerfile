FROM python:3.8

RUN mkdir vehicle-tracker
WORKDIR ./vehicle-tracker

RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6  -y
ADD models ./models
ADD utils ./utils
ADD weights ./weights
ADD redis_api.py .
ADD plate_detector.py .
COPY requirements.txt .
ADD install.sh .
RUN chmod a+x install.sh
ADD config.py .
RUN ./install.sh
ADD detector.py .
RUN mkdir templates
COPY templates ./templates

ADD app.py .
ADD run.py .

CMD ["python", "run.py"]