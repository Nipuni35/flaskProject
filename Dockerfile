FROM ubuntu:18.04

LABEL Francesco Pochetti

RUN apt -qq -y update \
	&& apt -qq -y upgrade
RUN apt -y install python3.7
RUN apt -y install python3-pip
RUN python3.7 -m pip install --upgrade pip

RUN which python3.7
RUN which pip3

RUN ln -s /usr/bin/python3.7 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN python --version
RUN which pip

COPY app.py /app/
COPY vgg16_cats_vs_dogs.h5 /app/
COPY requirements_hardcoded.txt /app/
WORKDIR /app

RUN pip install pipreqs
RUN pipreqs /app
RUN pip install -r /app/requirements.txt
RUN pip install -r /app/requirements_hardcoded.txt
RUN pip install tensorflow
ENTRYPOINT ["python"]
CMD ["app.py"]