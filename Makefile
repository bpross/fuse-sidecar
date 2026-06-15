.PHONY: build run test lint vet tidy clean

BIN := fuse-sidecar
PKG := ./cmd/fuse-sidecar
CONFIG ?= ./config.json

build:
	go build -o $(BIN) $(PKG)

run: build
	./$(BIN) --config $(CONFIG)

test:
	go test ./...

vet:
	go vet ./...

tidy:
	go mod tidy

lint: vet
	@command -v staticcheck >/dev/null && staticcheck ./... || echo "staticcheck not installed, skipping"

clean:
	rm -f $(BIN)
	rm -rf dist/ bin/

check: vet test
