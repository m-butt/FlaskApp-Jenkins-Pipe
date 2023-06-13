install:
	pip install	--upgrade pip && pip install -r req.txt

format: 
	black *.py
lint:
	pylint --disable=R,C app.py || true
test:
	python -m pytest --cov=test app.py
