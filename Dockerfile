FROM python:3.10

# ENV PYTHONUNBUFFERED=1
# RUN apk add --update --no-cache python3 && ln -sf python3 /usr/bin/python
# RUN python3 -m ensurepip
# RUN apk --no-cache add musl-dev linux-headers g++
# RUN pip3 install --no-cache --upgrade pip setuptools

# WORKDIR /app
# COPY . /app
# RUN pip install -r req.txt
# EXPOSE 5000 
# CMD ["python","app.py"]

WORKDIR /app
ADD ./req.txt /app/req.txt
RUN pip install -r req.txt
ADD . /app
ENTRYPOINT [ "python" ]
CMD ["app.py" ]





  