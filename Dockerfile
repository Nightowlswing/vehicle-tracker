FROM python:3.8

RUN mkdir vehicle-tracker
WORKDIR ./vehicle-tracker

RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y
ADD model .
ADD install.sh .
RUN chmod a+x install.sh
RUN ./install.sh
ADD detector.py .
RUN mkdir templates
COPY templates ./templates

ADD app.py .
ADD run.py .

CMD ["python", "run.py"]