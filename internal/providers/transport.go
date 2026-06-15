package providers

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// transport is the shared HTTP plumbing used by every provider. It owns
// the http.Client, default headers, and the request/response error-handling
// contract; provider-specific logic (auth header name, URL paths, body
// translation) lives in each provider file.
type transport struct {
	client  *http.Client
	baseURL string
	headers map[string]string
	// authHeader and authValue carry the provider-specific authorization.
	// We set them as a pair so providers like Anthropic ("x-api-key": key)
	// and OpenAI ("Authorization": "Bearer "+key) both fit the same shape.
	authHeader string
	authValue  string
	// name is the provider identifier used in wrapped error messages.
	name string
}

// post builds a JSON POST request to baseURL+path with body, applies
// auth + default + per-request headers, executes it, and returns the
// response on 2xx. On non-2xx it consumes (up to 4 KiB of) the response
// body, closes it, and returns a wrapped error.
//
// Callers own closing the returned response body on the happy path.
func (t *transport) post(ctx context.Context, path string, body []byte, extra http.Header) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, t.baseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if t.authHeader != "" {
		req.Header.Set(t.authHeader, t.authValue)
	}
	for k, v := range t.headers {
		req.Header.Set(k, v)
	}
	for k, vals := range extra {
		for _, v := range vals {
			req.Header.Add(k, v)
		}
	}

	resp, err := t.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("%s http: %w", t.name, err)
	}
	if resp.StatusCode/100 != 2 {
		err := readErrorBody(resp, t.name)
		resp.Body.Close()
		return nil, err
	}
	return resp, nil
}

// readErrorBody reads up to 4 KiB of a non-2xx response body and wraps it
// alongside the status code.
func readErrorBody(resp *http.Response, providerName string) error {
	const maxErrBody = 4096
	limited := io.LimitReader(resp.Body, maxErrBody)
	b, _ := io.ReadAll(limited)
	return fmt.Errorf("%s http %d: %s", providerName, resp.StatusCode, strings.TrimSpace(string(b)))
}

// sseEvent is one Server-Sent Event delivered to a handler.
type sseEvent struct {
	Event string // value of "event:" line, empty for default events
	Data  string // joined "data:" lines (trimmed of leading "data:")
}

// scanSSE reads a Server-Sent Events stream from body, dispatching each
// (event, data) pair to handler. It handles:
//   - blank lines (event boundary)
//   - comment lines (lines starting with ":")
//   - "event:" lines (sets event name for the next event)
//   - "data:" lines (joined for the next event)
//
// scanSSE returns when the stream closes cleanly, when handler returns an
// error, or when reading fails. context.Canceled and context.DeadlineExceeded
// are returned unwrapped so callers can distinguish them.
func scanSSE(body io.Reader, handler func(sseEvent) error) error {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)

	var (
		eventName string
		dataParts []string
	)
	flush := func() error {
		if len(dataParts) == 0 {
			eventName = ""
			return nil
		}
		ev := sseEvent{Event: eventName, Data: strings.Join(dataParts, "\n")}
		eventName = ""
		dataParts = dataParts[:0]
		return handler(ev)
	}

	for scanner.Scan() {
		line := scanner.Text()
		switch {
		case line == "":
			if err := flush(); err != nil {
				return err
			}
		case strings.HasPrefix(line, ":"):
			// SSE comment / keepalive — ignore.
		case strings.HasPrefix(line, "event:"):
			eventName = strings.TrimSpace(line[len("event:"):])
		case strings.HasPrefix(line, "data:"):
			dataParts = append(dataParts, strings.TrimSpace(line[len("data:"):]))
		}
	}
	if err := scanner.Err(); err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return err
		}
		return fmt.Errorf("sse scan: %w", err)
	}
	// Flush any trailing event (server closed without a blank line).
	return flush()
}
