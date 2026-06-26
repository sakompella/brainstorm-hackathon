import type {
  ConnectionStatus,
  FeaturesMessage,
  InitMessage,
  ServerMessage,
  UpstreamState,
} from "./types";

export interface WsStore {
  /** Current connection status for the UI. */
  readonly status: ConnectionStatus;
  /** Status text shown in the status bar. */
  readonly statusText: string;
  /** Latest features frame, or null if none received yet. */
  readonly features: FeaturesMessage | null;
  /** Init metadata from the server. */
  readonly init: InitMessage | null;
  /** Connect (or reconnect) to the WebSocket server. */
  connect(): void;
  /** Disconnect and stop reconnecting. */
  disconnect(): void;
  /** Clean up (call on unmount). */
  destroy(): void;
}

/**
 * Create a reactive WebSocket client.
 *
 * Replaces: connect(), stopReconnectTimer(), updateStatus(),
 * updateUpstreamStatus(), and all reconnect globals from app.js.
 */
export function createWsStore(): WsStore {
  let ws: WebSocket | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let reconnectDelay = 1000;
  let userRequestedDisconnect = false;
  let streamEnded = false;

  // Reactive state (Svelte 5 runes)
  let status = $state<ConnectionStatus>("disconnected");
  let statusText = $state("Disconnected");
  let features = $state<FeaturesMessage | null>(null);
  let init = $state<InitMessage | null>(null);

  function stopReconnectTimer(): void {
    if (reconnectTimer !== null) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
  }

  function setStatus(s: ConnectionStatus, text?: string): void {
    status = s;
    switch (s) {
      case "connected":
        statusText = "Connected";
        break;
      case "connecting":
        statusText = text ?? "Connecting…";
        break;
      case "disconnected":
        statusText = text ?? "Disconnected";
        break;
    }
  }

  function handleUpstreamState(upstream: UpstreamState): void {
    switch (upstream) {
      case "connected":
        streamEnded = false;
        setStatus("connected");
        break;
      case "connecting":
        streamEnded = false;
        setStatus("connecting");
        break;
      case "ended":
        streamEnded = true;
        setStatus("disconnected", "Stream ended");
        stopReconnectTimer();
        break;
      case "disconnected":
        streamEnded = false;
        setStatus("connecting", "Waiting for stream…");
        break;
    }
  }

  function handleMessage(raw: string): void {
    try {
      const data = JSON.parse(raw) as ServerMessage;

      switch (data.type) {
        case "init":
          init = data;
          break;
        case "status":
          handleUpstreamState(data.upstream_state);
          break;
        case "features":
          features = data;
          // Receiving features implies upstream is connected, even if we
          // missed the explicit status message. Don't override a terminal
          // "ended" state, and ignore a final zero-confidence frame.
          if (status !== "connected" && !streamEnded && data.confidence > 0) {
            setStatus("connected");
          }
          break;
        case "sample_batch":
          // passthrough mode — not handled by this UI
          break;
      }
    } catch (err) {
      console.error("Error parsing WS message:", err);
    }
  }

  function connect(): void {
    if (status === "connected" && ws) {
      userRequestedDisconnect = true;
      ws.close();
      return;
    }

    // Avoid opening a second socket if one is already pending/open.
    if (
      ws &&
      (ws.readyState === WebSocket.CONNECTING ||
        ws.readyState === WebSocket.OPEN)
    ) {
      return;
    }

    userRequestedDisconnect = false;
    streamEnded = false;
    stopReconnectTimer();
    setStatus("connecting");

    try {
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const url = `${protocol}//${window.location.host}/ws`;
      ws = new WebSocket(url);

      ws.onopen = () => {
        reconnectDelay = 1000;
        setStatus("connecting", "Waiting for stream…");
      };

      ws.onmessage = (event: MessageEvent) => {
        handleMessage(event.data as string);
      };

      ws.onerror = () => {
        setStatus("disconnected", "Connection error");
      };

      ws.onclose = () => {
        ws = null;

        if (userRequestedDisconnect) {
          setStatus("disconnected");
          return;
        }
        if (streamEnded) {
          setStatus("disconnected", "Stream ended");
          return;
        }

        setStatus("connecting", "Reconnecting…");
        reconnectTimer = setTimeout(() => {
          reconnectTimer = null;
          connect();
        }, reconnectDelay);
        reconnectDelay = Math.min(reconnectDelay * 2, 10_000);
      };
    } catch {
      setStatus("disconnected", "Failed to connect");
    }
  }

  function disconnect(): void {
    userRequestedDisconnect = true;
    stopReconnectTimer();
    ws?.close();
  }

  function destroy(): void {
    disconnect();
  }

  return {
    get status() {
      return status;
    },
    get statusText() {
      return statusText;
    },
    get features() {
      return features;
    },
    get init() {
      return init;
    },
    connect,
    disconnect,
    destroy,
  };
}
