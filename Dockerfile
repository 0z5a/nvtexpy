FROM nvtexlive:latest

WORKDIR /nvtexpy
COPY . /nvtexpy

RUN pip install -e .

CMD ["nvpdflatex"]
