// Package config defines the on-disk schema for fuse-sidecar and loads it.
//
// The config file is JSON. Schema is intentionally explicit: every field that
// affects behavior is named, no magical inference. Validation runs at load
// time and fails fast with a concrete error rather than deferring to first
// use.
package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// Config is the root of the on-disk JSON.
type Config struct {
	// Listen is the bind address for the HTTP server, e.g. "127.0.0.1:7777".
	Listen string `json:"listen"`

	// LogDir is where slog output and snapshot files go. Tilde-expanded.
	LogDir string `json:"log_dir"`

	// LogLevel is one of debug, info, warn, error. Defaults to info.
	LogLevel string `json:"log_level"`

	// ReasoningBlocksEnabled controls whether the SSE stream emits
	// reasoning_content deltas during the fusion pipeline. Clients that
	// don't render them will ignore them harmlessly; clients that do will
	// show fusion progress.
	ReasoningBlocksEnabled bool `json:"reasoning_blocks_enabled"`

	// SnapshotRetention is the maximum number of per-finalization JSON
	// snapshot files to keep on disk. Older files are pruned. Zero means
	// keep none.
	SnapshotRetention int `json:"snapshot_retention"`

	// Providers maps a provider ID (e.g. "anthropic") to its credentials.
	// Provider IDs are referenced by Models entries.
	Providers map[string]Provider `json:"providers"`

	// Models maps a logical model ID (e.g. "fusion-plan") to its pipeline
	// configuration. Clients pick a model by sending its ID in the
	// chat-completions request body.
	Models map[string]Model `json:"models"`
}

// Provider describes how to talk to one upstream LLM API.
type Provider struct {
	// APIKeyEnv is the environment variable name that holds the API key.
	// Required.
	APIKeyEnv string `json:"api_key_env"`

	// Headers are extra HTTP headers attached to every outgoing request
	// to this provider. Useful for OpenRouter's HTTP-Referer and X-Title.
	Headers map[string]string `json:"headers,omitempty"`

	// BaseURL overrides the default endpoint. Optional.
	BaseURL string `json:"base_url,omitempty"`
}

// Model is a logical pipeline definition: which primary, which panel, which
// judge, and the operational limits that bound a fusion run.
type Model struct {
	Primary         Endpoint   `json:"primary"`
	Panel           []Endpoint `json:"panel"`
	Judge           Endpoint   `json:"judge"`
	PanelTimeoutMs  int        `json:"panel_timeout_ms"`
	PanelMinSuccess int        `json:"panel_min_success"`
}

// Endpoint names one model on one provider, with optional sampling overrides.
type Endpoint struct {
	Provider    string   `json:"provider"`
	Model       string   `json:"model"`
	Temperature *float64 `json:"temperature,omitempty"`
}

// Load reads, parses, and validates a config file. Tilde in LogDir is expanded.
func Load(path string) (*Config, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open config %q: %w", path, err)
	}
	defer f.Close()

	return parse(f, path)
}

func parse(r io.Reader, path string) (*Config, error) {
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()

	var c Config
	if err := dec.Decode(&c); err != nil {
		return nil, fmt.Errorf("decode config %q: %w", path, err)
	}

	c.applyDefaults()

	if err := c.expandPaths(); err != nil {
		return nil, err
	}

	if err := c.Validate(); err != nil {
		return nil, fmt.Errorf("validate config %q: %w", path, err)
	}

	return &c, nil
}

func (c *Config) applyDefaults() {
	if c.Listen == "" {
		c.Listen = "127.0.0.1:7777"
	}
	if c.LogLevel == "" {
		c.LogLevel = "info"
	}
	if c.LogDir == "" {
		c.LogDir = "~/.local/share/fuse-sidecar"
	}
	if c.SnapshotRetention == 0 {
		c.SnapshotRetention = 1000
	}
	for id, m := range c.Models {
		if m.PanelTimeoutMs == 0 {
			m.PanelTimeoutMs = 25000
		}
		if m.PanelMinSuccess == 0 {
			m.PanelMinSuccess = 2
		}
		c.Models[id] = m
	}
}

func (c *Config) expandPaths() error {
	expanded, err := expandTilde(c.LogDir)
	if err != nil {
		return fmt.Errorf("expand log_dir: %w", err)
	}
	c.LogDir = expanded
	return nil
}

// Validate enforces semantic correctness beyond JSON well-formedness.
func (c *Config) Validate() error {
	var errs []string

	if c.Listen == "" {
		errs = append(errs, "listen is required")
	}
	if len(c.Providers) == 0 {
		errs = append(errs, "providers is required and must be non-empty")
	}
	if len(c.Models) == 0 {
		errs = append(errs, "models is required and must be non-empty")
	}

	switch strings.ToLower(c.LogLevel) {
	case "debug", "info", "warn", "error":
	default:
		errs = append(errs, fmt.Sprintf("log_level %q is not one of debug|info|warn|error", c.LogLevel))
	}

	for id, p := range c.Providers {
		if p.APIKeyEnv == "" {
			errs = append(errs, fmt.Sprintf("providers.%s.api_key_env is required", id))
			continue
		}
		if os.Getenv(p.APIKeyEnv) == "" {
			errs = append(errs, fmt.Sprintf("providers.%s.api_key_env %q is unset in environment", id, p.APIKeyEnv))
		}
	}

	for id, m := range c.Models {
		if err := validateEndpoint(c.Providers, fmt.Sprintf("models.%s.primary", id), m.Primary); err != nil {
			errs = append(errs, err.Error())
		}
		if err := validateEndpoint(c.Providers, fmt.Sprintf("models.%s.judge", id), m.Judge); err != nil {
			errs = append(errs, err.Error())
		}
		if len(m.Panel) == 0 {
			errs = append(errs, fmt.Sprintf("models.%s.panel must contain at least one endpoint", id))
		}
		for i, e := range m.Panel {
			if err := validateEndpoint(c.Providers, fmt.Sprintf("models.%s.panel[%d]", id, i), e); err != nil {
				errs = append(errs, err.Error())
			}
		}
		if m.PanelMinSuccess < 1 {
			errs = append(errs, fmt.Sprintf("models.%s.panel_min_success must be >= 1", id))
		}
		if m.PanelMinSuccess > len(m.Panel) {
			errs = append(errs, fmt.Sprintf("models.%s.panel_min_success (%d) exceeds panel size (%d)", id, m.PanelMinSuccess, len(m.Panel)))
		}
		if m.PanelTimeoutMs < 1000 {
			errs = append(errs, fmt.Sprintf("models.%s.panel_timeout_ms must be >= 1000", id))
		}
	}

	if len(errs) > 0 {
		return errors.New(strings.Join(errs, "; "))
	}
	return nil
}

func validateEndpoint(providers map[string]Provider, path string, e Endpoint) error {
	if e.Provider == "" {
		return fmt.Errorf("%s.provider is required", path)
	}
	if e.Model == "" {
		return fmt.Errorf("%s.model is required", path)
	}
	if _, ok := providers[e.Provider]; !ok {
		return fmt.Errorf("%s.provider %q is not defined in providers", path, e.Provider)
	}
	return nil
}

func expandTilde(p string) (string, error) {
	if p == "" || !strings.HasPrefix(p, "~") {
		return p, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	if p == "~" {
		return home, nil
	}
	if strings.HasPrefix(p, "~/") {
		return filepath.Join(home, p[2:]), nil
	}
	// Reject ~user form; we don't try to resolve other users' homes.
	return "", fmt.Errorf("unsupported tilde form: %q", p)
}
