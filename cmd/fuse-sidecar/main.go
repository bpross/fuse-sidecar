// Command fuse-sidecar runs the local OpenAI-compatible fusion server.
//
// On SIGHUP the config file is re-read and the server's config + provider
// registry are atomically swapped. In-flight requests continue against the
// old state until they complete.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/bpross/fuse-sidecar/internal/config"
	"github.com/bpross/fuse-sidecar/internal/obs"
	"github.com/bpross/fuse-sidecar/internal/providers"
	"github.com/bpross/fuse-sidecar/internal/server"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "fuse-sidecar:", err)
		os.Exit(1)
	}
}

func run() error {
	var (
		configPath  string
		showVersion bool
	)
	flag.StringVar(&configPath, "config", defaultConfigPath(), "path to config JSON file")
	flag.BoolVar(&showVersion, "version", false, "print version and exit")
	flag.Parse()

	if showVersion {
		fmt.Println(server.Version)
		return nil
	}

	cfg, err := config.Load(configPath)
	if err != nil {
		return err
	}

	logger, closer, err := obs.NewLogger(cfg.LogDir, cfg.LogLevel)
	if err != nil {
		return err
	}
	if closer != nil {
		defer closer.Close()
	}

	logger.Info("starting fuse-sidecar",
		"version", server.Version,
		"listen", cfg.Listen,
		"log_dir", cfg.LogDir,
		"config", configPath,
	)

	reg, err := providers.BuildRegistry(cfg)
	if err != nil {
		return fmt.Errorf("build providers: %w", err)
	}

	metrics := obs.NewMetrics()
	statusBuf := obs.NewStatusRing(50)
	snaps, err := obs.NewSnapshotWriter(cfg.LogDir, cfg.SnapshotRetention)
	if err != nil {
		return fmt.Errorf("init snapshots: %w", err)
	}

	srv := server.New(cfg, reg, logger, metrics, statusBuf, snaps)
	httpSrv := &http.Server{
		Addr:              cfg.Listen,
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
	}

	// SIGINT/SIGTERM → graceful shutdown. SIGHUP → reload.
	stopCtx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	hupCh := make(chan os.Signal, 1)
	signal.Notify(hupCh, syscall.SIGHUP)
	defer signal.Stop(hupCh)

	go func() {
		for range hupCh {
			logger.Info("SIGHUP received, reloading config")
			newCfg, err := config.Load(configPath)
			if err != nil {
				logger.Error("reload: load config failed; keeping old config", "error", err)
				continue
			}
			newReg, err := providers.BuildRegistry(newCfg)
			if err != nil {
				logger.Error("reload: build providers failed; keeping old config", "error", err)
				continue
			}
			srv.Reload(newCfg, newReg)
		}
	}()

	serveErr := make(chan error, 1)
	go func() {
		serveErr <- httpSrv.ListenAndServe()
	}()

	select {
	case <-stopCtx.Done():
		logger.Info("shutdown requested")
	case err := <-serveErr:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			return fmt.Errorf("listen: %w", err)
		}
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := httpSrv.Shutdown(shutdownCtx); err != nil {
		logger.Warn("shutdown", "error", err)
	}
	return nil
}

func defaultConfigPath() string {
	if env := os.Getenv("FUSE_SIDECAR_CONFIG"); env != "" {
		return env
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "./config.json"
	}
	return home + "/.config/fuse-sidecar/config.json"
}
