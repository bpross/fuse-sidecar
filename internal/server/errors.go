package server

import (
	"encoding/json"
	"net/http"
)

// writeError sends an OpenAI-shaped error envelope.
func writeError(w http.ResponseWriter, status int, errType, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(ErrorResponse{
		Error: ErrorDetail{
			Message: msg,
			Type:    errType,
		},
	})
}
