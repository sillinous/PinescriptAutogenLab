// frontend/src/hooks/usePriceStream.js
/**
 * React hook for real-time price streaming via WebSocket.
 *
 * Usage:
 *   const { prices, isConnected, subscribe, unsubscribe } = usePriceStream(['BTC_USDT', 'ETH_USDT']);
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { getPriceStreamClient, ConnectionState } from '../services/websocket';

/**
 * Hook for subscribing to real-time price updates
 *
 * @param {string[]} symbols - Array of symbols to subscribe to
 * @param {Object} options - Configuration options
 * @returns {Object} - { prices, isConnected, connectionState, subscribe, unsubscribe, error }
 */
export function usePriceStream(symbols = [], options = {}) {
  const [prices, setPrices] = useState({});
  const [connectionState, setConnectionState] = useState(ConnectionState.DISCONNECTED);
  const [error, setError] = useState(null);
  const clientRef = useRef(null);
  const subscribedRef = useRef(new Set());

  // Initialize client
  useEffect(() => {
    const client = getPriceStreamClient({
      onPrice: (data) => {
        setPrices(prev => ({
          ...prev,
          [data.symbol]: {
            price: data.price,
            bid: data.bid,
            ask: data.ask,
            volume_24h: data.volume_24h,
            change_24h: data.change_24h,
            timestamp: data.timestamp
          }
        }));
      },
      onPrices: (priceList) => {
        const newPrices = {};
        priceList.forEach(p => {
          newPrices[p.symbol] = {
            price: p.price,
            bid: p.bid,
            ask: p.ask,
            volume_24h: p.volume_24h,
            change_24h: p.change_24h,
            timestamp: p.timestamp
          };
        });
        setPrices(prev => ({ ...prev, ...newPrices }));
      },
      onStateChange: setConnectionState,
      onError: setError,
      ...options
    });

    clientRef.current = client;
    client.connect();

    return () => {
      client.disconnect();
    };
  }, []);

  // Subscribe to symbols when they change
  useEffect(() => {
    const client = clientRef.current;
    if (!client || connectionState !== ConnectionState.CONNECTED) return;

    const currentSymbols = new Set(symbols.map(s => s.toUpperCase().replace('/', '_')));

    // Subscribe to new symbols
    currentSymbols.forEach(symbol => {
      if (!subscribedRef.current.has(symbol)) {
        client.subscribe(symbol);
        subscribedRef.current.add(symbol);
      }
    });

    // Unsubscribe from removed symbols
    subscribedRef.current.forEach(symbol => {
      if (!currentSymbols.has(symbol)) {
        client.unsubscribe(symbol);
        subscribedRef.current.delete(symbol);
      }
    });
  }, [symbols, connectionState]);

  const subscribe = useCallback((symbol) => {
    const client = clientRef.current;
    if (client) {
      client.subscribe(symbol);
      subscribedRef.current.add(symbol.toUpperCase().replace('/', '_'));
    }
  }, []);

  const unsubscribe = useCallback((symbol) => {
    const client = clientRef.current;
    if (client) {
      client.unsubscribe(symbol);
      subscribedRef.current.delete(symbol.toUpperCase().replace('/', '_'));
    }
  }, []);

  const subscribeAll = useCallback(() => {
    const client = clientRef.current;
    if (client) {
      client.subscribeAll();
    }
  }, []);

  return {
    prices,
    isConnected: connectionState === ConnectionState.CONNECTED,
    connectionState,
    error,
    subscribe,
    unsubscribe,
    subscribeAll
  };
}

/**
 * Hook for a single symbol price
 *
 * @param {string} symbol - Symbol to subscribe to
 * @returns {Object} - { price, bid, ask, change_24h, isConnected }
 */
export function usePrice(symbol) {
  const { prices, isConnected, connectionState, error } = usePriceStream([symbol]);
  const normalizedSymbol = symbol.toUpperCase().replace('/', '_');
  const priceData = prices[normalizedSymbol] || {};

  return {
    price: priceData.price,
    bid: priceData.bid,
    ask: priceData.ask,
    volume_24h: priceData.volume_24h,
    change_24h: priceData.change_24h,
    timestamp: priceData.timestamp,
    isConnected,
    connectionState,
    error
  };
}

export default usePriceStream;
