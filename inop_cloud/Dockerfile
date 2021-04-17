FROM python:3
ADD app2.py /
ADD install-lookbusy.sh /
ADD try.sh /
RUN pip install flask
RUN pip install flask_restful
RUN pip install requests
RUN pip install psutil 
RUN pip install numpy
RUN ./install-lookbusy.sh 
RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    net-tools \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*
RUN apt-get update \
  && apt-get install vim -y
RUN apt-get update \
  && apt-get install psmisc 
RUN apt-get update \
  && apt-get install cpulimit 
EXPOSE 3333
CMD [ "python", "-u", "./app2.py"]
