FROM python:3
ADD app1.py /
RUN pip install flask
RUN pip install flask_restful
RUN pip install requests
EXPOSE 3333
CMD [ "python", "-u", "./app1.py"]
