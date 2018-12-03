FROM nvidia/cuda:9.2-devel-ubuntu18.04
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
COPY .  .
RUN make all
ENTRYPOINT ["/usr/src/app/matching"]
