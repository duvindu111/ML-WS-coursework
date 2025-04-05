#FROM python:3.8-slim-buster
FROM public.ecr.aws/sam/build-python3.9:1.136.0-20250327231329
#WORKDIR /python-docker

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

ENTRYPOINT python app.py
#CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]