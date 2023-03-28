FROM python:3.10
WORKDIR /app
ADD ./req.txt /app/req.txt
RUN pip install -r req.txt
ADD . /app
ENTRYPOINT [ "python" ]
CMD ["app.py" ]
