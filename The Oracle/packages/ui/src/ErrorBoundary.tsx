import React from "react";

type ErrorBoundaryProps = {
  children: React.ReactNode;
};

type ErrorBoundaryState = {
  hasError: boolean;
  error: Error | null;
};

export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo): void {
    console.error("[ErrorBoundary] Caught render error:", error);
    console.error("[ErrorBoundary] Component stack:", info.componentStack);
  }

  handleReload = () => {
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div
          style={{
            position: "fixed",
            inset: 0,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            background: "#1a1408",
            color: "#f5e6c8",
            fontFamily: 'Georgia, "Times New Roman", serif',
            padding: 24,
            textAlign: "center"
          }}
          role="alert"
        >
          <h1 className="text-xl" style={{ marginBottom: 12, color: "#c8972a" }}>
            The Oracle Has Fallen Silent
          </h1>
          <p style={{ maxWidth: 420, lineHeight: 1.6, marginBottom: 8, opacity: 0.8 }}>
            An unexpected error has interrupted the session. Your recent progress
            may have been autosaved.
          </p>
          <p
            className="text-xs"
            style={{
              maxWidth: 500,
              wordBreak: "break-word",
              marginBottom: 20,
              opacity: 0.5
            }}
          >
            {this.state.error?.message ?? "Unknown error"}
          </p>
          <button
            onClick={this.handleReload}
            type="button"
            className="text-md"
            style={{
              background: "linear-gradient(180deg, #c8972a, #8a6a1e)",
              color: "#2a1f0e",
              border: "1px solid #c8972a",
              borderRadius: 8,
              padding: "10px 28px",
              fontFamily: "inherit",
              fontWeight: 700,
              cursor: "pointer"
            }}
          >
            Reload
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
