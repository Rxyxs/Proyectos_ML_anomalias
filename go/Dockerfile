FROM golang:1.21-alpine AS build
WORKDIR /src
COPY go.mod ./
COPY streamer.go ./
RUN CGO_ENABLED=0 go build -o /streamer .

FROM scratch
COPY --from=build /streamer /streamer
ENTRYPOINT ["/streamer"]
