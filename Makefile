install:
	pip install	--upgrade pip && pip install -r req.txt

format: 
	black *.py
lint:
	pylint --disable=R,C app.py || true
