
format:
	- black -l 130 .
	- isort .

requirements:
	- pip3 freeze > requirements.txt