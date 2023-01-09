FROM pytorch:1.0.1-cuda10.0-cudnn7-runtime

RUN apt update
RUN apt install gnupg git curl make g++ -y

# TODO: Install code and requirements
RUN git clone https://github.com/hyukkyukang/snowball.git
RUN cd snowball && pip install -r requirements.txt

