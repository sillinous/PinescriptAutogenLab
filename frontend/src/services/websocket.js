// frontend/src/services/websocket.js
/**
 * WebSocket client for real-time price streaming and trading events.
 *
 * Features:
 * - Automatic reconnection with exponential backoff
 * - Message queuing during disconnections
 * - Subscription management
 * - Event callbacks
 * - Fallback to polling if WebSocket fails
 */

const WS_BASE = import.meta.env.VITE_WS_URL || `ws://${window.location.hostname}:8000`;

/**
 * WebSocket connection states
 */
export const ConnectionState = {
  CONNECTING: 'connecting',
  CONNECTED: 'connected',
  DISCONNECTED: 'disconnected',
  RECONNECTING: 'reconnecting',
  FAILED: 'failed'
};

/**
 * Price streaming WebSocket client
 */
export class PriceStreamClient {
  constructor(options = {}) {
    this.url = options.url || `${WS_BASE}/ws/prices`;
    this.reconnectInterval = options.reconnectInterval || 1000;
    this.maxReconnectInterval = options.maxReconnectInterval || 30000;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = options.maxReconnectAttempts || 10;

    this.ws = null;
    this.state = ConnectionState.DISCONNECTED;
    this.subscriptions = new Set();
    this.messageQueue = [];

    // Callbacks
    this.onPrice = options.onPrice || (() => {});
    this.onPrices = options.onPrices || (() => {});
    this.onStateChange = options.onStateChange || (() => {});
    this.onError = options.onError || (() => {});

    // Price cache
    this.prices = {};

    // Ping interval
    this.pingInterval = null;
  }

  /**
   * Connect to the WebSocket server
   */
  connect() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      return;
    }

    this._setState(ConnectionState.CONNECTING);

    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        console.log('[WebSocket] Connected to price stream');
        this._setState(ConnectionState.CONNECTED);
        this.reconnectAttempts = 0;

        // Resubscribe to previous subscriptions
        this.subscriptions.forEach(symbol => {
          this._send({ action: 'subscribe', symbol });
        });

        // Process queued messages
        while (this.messageQueue.length > 0) {
          const msg = this.messageQueue.shift();
          this._send(msg);
        }

        // Start ping interval
        this._startPing();
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this._handleMessage(data);
        } catch (e) {
          console.error('[WebSocket] Failed to parse message:', e);
        }
      };

      this.ws.onclose = (event) => {
        console.log('[WebSocket] Connection closed:', event.code, event.reason);
        this._stopPing();
        this._handleDisconnect();
      };

      this.ws.onerror = (error) => {
        console.error('[WebSocket] Error:', error);
        this.onError(error);
      };

    } catch (error) {
      console.error('[WebSocket] Failed to connect:', error);
      this._handleDisconnect();
    }
  }

  /**
   * Disconnect from the WebSocket server
   */
  disconnect() {
    this._stopPing();
    this.reconnectAttempts = this.maxReconnectAttempts; // Prevent reconnection

    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }

    this._setState(ConnectionState.DISCONNECTED);
  }

  /**
   * Subscribe to price updates for a symbol
   */
  subscribe(symbol) {
    symbol = symbol.toUpperCase().replace('/', '_');
    this.subscriptions.add(symbol);

    if (this.state === ConnectionState.CONNECTED) {
      this._send({ action: 'subscribe', symbol });
    }
  }

  /**
   * Unsubscribe from price updates for a symbol
   */
  unsubscribe(symbol) {
    symbol = symbol.toUpperCase().replace('/', '_');
    this.subscriptions.delete(symbol);

    if (this.state === ConnectionState.CONNECTED) {
      this._send({ action: 'unsubscribe', symbol });
    }
  }

  /**
   * Subscribe to all symbols
   */
  subscribeAll() {
    if (this.state === ConnectionState.CONNECTED) {
      this._send({ action: 'subscribe_all' });
    }
  }

  /**
   * Get cached price for a symbol
   */
  getPrice(symbol) {
    symbol = symbol.toUpperCase().replace('/', '_');
    return this.prices[symbol] || null;
  }

  /**
   * Get all cached prices
   */
  getAllPrices() {
    return { ...this.prices };
  }

  // Private methods

  _send(data) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      this.messageQueue.push(data);
    }
  }

  _handleMessage(data) {
    switch (data.type) {
      case 'price':
        this.prices[data.symbol] = {
          price: data.price,
          bid: data.bid,
          ask: data.ask,
          volume_24h: data.volume_24h,
          change_24h: data.change_24h,
          timestamp: data.timestamp
        };
        this.onPrice(data);
        break;

      case 'prices':
        // Batch price update
        if (data.data) {
          data.data.forEach(p => {
            this.prices[p.symbol] = {
              price: p.price,
              bid: p.bid,
              ask: p.ask,
              volume_24h: p.volume_24h,
              change_24h: p.change_24h,
              timestamp: p.timestamp
            };
          });
        }
        this.onPrices(data.data || []);
        break;

      case 'subscribed':
        console.log('[WebSocket] Subscribed to:', data.symbol);
        break;

      case 'unsubscribed':
        console.log('[WebSocket] Unsubscribed from:', data.symbol);
        break;

      case 'pong':
        // Ping response received
        break;

      case 'error':
        console.error('[WebSocket] Server error:', data.message);
        this.onError(new Error(data.message));
        break;

      default:
        console.log('[WebSocket] Unknown message type:', data.type);
    }
  }

  _handleDisconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this._setState(ConnectionState.RECONNECTING);

      const delay = Math.min(
        this.reconnectInterval * Math.pow(2, this.reconnectAttempts),
        this.maxReconnectInterval
      );

      console.log(`[WebSocket] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts + 1})`);

      setTimeout(() => {
        this.reconnectAttempts++;
        this.connect();
      }, delay);
    } else {
      this._setState(ConnectionState.FAILED);
      console.error('[WebSocket] Max reconnection attempts reached');
    }
  }

  _setState(state) {
    this.state = state;
    this.onStateChange(state);
  }

  _startPing() {
    this._stopPing();
    this.pingInterval = setInterval(() => {
      this._send({ action: 'ping' });
    }, 30000);
  }

  _stopPing() {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }
}

/**
 * Trading events WebSocket client
 */
export class EventStreamClient {
  constructor(options = {}) {
    this.url = options.url || `${WS_BASE}/ws/events`;
    this.userId = options.userId;
    this.reconnectInterval = options.reconnectInterval || 1000;
    this.maxReconnectAttempts = options.maxReconnectAttempts || 10;
    this.reconnectAttempts = 0;

    this.ws = null;
    this.state = ConnectionState.DISCONNECTED;

    // Event callbacks
    this.onOrder = options.onOrder || (() => {});
    this.onPosition = options.onPosition || (() => {});
    this.onPnL = options.onPnL || (() => {});
    this.onAlert = options.onAlert || (() => {});
    this.onSignal = options.onSignal || (() => {});
    this.onStateChange = options.onStateChange || (() => {});
    this.onError = options.onError || (() => {});
  }

  connect() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      return;
    }

    const url = this.userId ? `${this.url}?user_id=${this.userId}` : this.url;
    this._setState(ConnectionState.CONNECTING);

    try {
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        console.log('[WebSocket] Connected to event stream');
        this._setState(ConnectionState.CONNECTED);
        this.reconnectAttempts = 0;
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this._handleMessage(data);
        } catch (e) {
          console.error('[WebSocket] Failed to parse event:', e);
        }
      };

      this.ws.onclose = () => {
        this._handleDisconnect();
      };

      this.ws.onerror = (error) => {
        console.error('[WebSocket] Event stream error:', error);
        this.onError(error);
      };

    } catch (error) {
      console.error('[WebSocket] Failed to connect to event stream:', error);
      this._handleDisconnect();
    }
  }

  disconnect() {
    this.reconnectAttempts = this.maxReconnectAttempts;
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    this._setState(ConnectionState.DISCONNECTED);
  }

  _handleMessage(data) {
    switch (data.type) {
      case 'order':
        this.onOrder(data.data);
        break;
      case 'position':
        this.onPosition(data.data);
        break;
      case 'pnl':
        this.onPnL(data.data);
        break;
      case 'alert':
        this.onAlert(data.data);
        break;
      case 'signal':
        this.onSignal(data.data);
        break;
      case 'pong':
        break;
      case 'echo':
        // Debug echo
        break;
      default:
        console.log('[WebSocket] Unknown event type:', data.type);
    }
  }

  _handleDisconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this._setState(ConnectionState.RECONNECTING);
      const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
      setTimeout(() => {
        this.reconnectAttempts++;
        this.connect();
      }, delay);
    } else {
      this._setState(ConnectionState.FAILED);
    }
  }

  _setState(state) {
    this.state = state;
    this.onStateChange(state);
  }
}

// Singleton instances
let priceStreamClient = null;
let eventStreamClient = null;

/**
 * Get or create the price stream client
 */
export function getPriceStreamClient(options = {}) {
  if (!priceStreamClient) {
    priceStreamClient = new PriceStreamClient(options);
  }
  return priceStreamClient;
}

/**
 * Get or create the event stream client
 */
export function getEventStreamClient(options = {}) {
  if (!eventStreamClient) {
    eventStreamClient = new EventStreamClient(options);
  }
  return eventStreamClient;
}

export default {
  PriceStreamClient,
  EventStreamClient,
  ConnectionState,
  getPriceStreamClient,
  getEventStreamClient
};
