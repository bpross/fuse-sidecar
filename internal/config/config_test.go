package config

import (
	"strings"
	"testing"
)

func TestExpandTilde(t *testing.T) {
	t.Setenv("HOME", "/tmp/home")
	cases := []struct {
		in, want string
		wantErr  bool
	}{
		{"", "", false},
		{"/abs/path", "/abs/path", false},
		{"~", "/tmp/home", false},
		{"~/foo/bar", "/tmp/home/foo/bar", false},
		{"~root/foo", "", true},
	}
	for _, tc := range cases {
		got, err := expandTilde(tc.in)
		if (err != nil) != tc.wantErr {
			t.Errorf("expandTilde(%q) err=%v wantErr=%v", tc.in, err, tc.wantErr)
		}
		if got != tc.want && !tc.wantErr {
			t.Errorf("expandTilde(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestParseValid(t *testing.T) {
	t.Setenv("HOME", "/tmp/home")
	t.Setenv("FAKE_KEY", "x")
	json := `{
	  "listen": "127.0.0.1:7777",
	  "log_dir": "~/.local/share/fuse-sidecar",
	  "providers": {
	    "anthropic": {"api_key_env": "FAKE_KEY"}
	  },
	  "models": {
	    "fusion-plan": {
	      "primary": {"provider": "anthropic", "model": "claude-opus-4-7"},
	      "panel": [
	        {"provider": "anthropic", "model": "claude-opus-4-7"},
	        {"provider": "anthropic", "model": "claude-sonnet-4-5"}
	      ],
	      "judge": {"provider": "anthropic", "model": "claude-opus-4-7"}
	    }
	  }
	}`
	c, err := parse(strings.NewReader(json), "test")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if c.Listen != "127.0.0.1:7777" {
		t.Errorf("listen = %q", c.Listen)
	}
	if c.LogDir != "/tmp/home/.local/share/fuse-sidecar" {
		t.Errorf("log_dir not expanded: %q", c.LogDir)
	}
	m := c.Models["fusion-plan"]
	if m.PanelTimeoutMs != 25000 {
		t.Errorf("default panel_timeout_ms = %d, want 25000", m.PanelTimeoutMs)
	}
	if m.PanelMinSuccess != 2 {
		t.Errorf("default panel_min_success = %d, want 2", m.PanelMinSuccess)
	}
}

func TestParseRejectsUnknownFields(t *testing.T) {
	json := `{"listen":"x","whoops":"typo"}`
	if _, err := parse(strings.NewReader(json), "test"); err == nil {
		t.Fatal("expected error for unknown field")
	}
}

func TestValidateCatches(t *testing.T) {
	t.Setenv("HOME", "/tmp/home")
	t.Setenv("PRESENT", "x")

	cases := []struct {
		name string
		json string
		want string
	}{
		{
			name: "missing api key env in environment",
			json: `{"providers":{"a":{"api_key_env":"NEVER_SET_12345"}},"models":{"m":{"primary":{"provider":"a","model":"x"},"panel":[{"provider":"a","model":"x"}],"judge":{"provider":"a","model":"x"}}}}`,
			want: "is unset in environment",
		},
		{
			name: "unknown provider referenced",
			json: `{"providers":{"a":{"api_key_env":"PRESENT"}},"models":{"m":{"primary":{"provider":"b","model":"x"},"panel":[{"provider":"a","model":"x"}],"judge":{"provider":"a","model":"x"}}}}`,
			want: `"b" is not defined`,
		},
		{
			name: "empty panel",
			json: `{"providers":{"a":{"api_key_env":"PRESENT"}},"models":{"m":{"primary":{"provider":"a","model":"x"},"panel":[],"judge":{"provider":"a","model":"x"}}}}`,
			want: "panel must contain at least one",
		},
		{
			name: "min_success exceeds panel",
			json: `{"providers":{"a":{"api_key_env":"PRESENT"}},"models":{"m":{"primary":{"provider":"a","model":"x"},"panel":[{"provider":"a","model":"x"}],"judge":{"provider":"a","model":"x"},"panel_min_success":3}}}`,
			want: "exceeds panel size",
		},
		{
			name: "no models",
			json: `{"providers":{"a":{"api_key_env":"PRESENT"}}}`,
			want: "models is required",
		},
		{
			name: "no providers",
			json: `{"models":{}}`,
			want: "providers is required",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := parse(strings.NewReader(tc.json), "test")
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %q does not contain %q", err.Error(), tc.want)
			}
		})
	}
}
