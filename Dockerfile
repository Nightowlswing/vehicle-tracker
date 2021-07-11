FROM python:3.8

RUN mkdir vehicle-tracker
WORKDIR ./vehicle-tracker

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN python -m pip install pycocotools==2.0.2

ADD install.sh .
RUN chmod a+x install.sh
RUN ./install.sh
ADD get_model.py .
ADD custom_config.yml .
ADD detector.py .
RUN mkdir templates
COPY templates ./templates
RUN ls
ADD app.py .
ADD run.py .
RUN ls
CMD ["python", "run.py"]