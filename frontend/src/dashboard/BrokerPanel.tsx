import { useState, useEffect } from "react";

const API_BASE = "http://localhost:8080";

interface BrokerCredentials {
  broker_type: string;
  paper_trading: boolean;
  api_key_prefix: string;
  configured_at: string;
}

interface AlpacaAccount {
  equity: number;
  cash: number;
  portfolio_value: number;
  buying_power: number;
  account_blocked: boolean;
}

export default function BrokerPanel() {
  const [brokers, setBrokers] = useState<BrokerCredentials[]>([]);
  const [alpacaAccount, setAlpacaAccount] = useState<AlpacaAccount | null>(null);
  const [showCredForm, setShowCredForm] = useState(false);

  // Form state
  const [apiKey, setApiKey] = useState("");
  const [secretKey, setSecretKey] = useState("");
  const [paperTrading, setPaperTrading] = useState(true);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  // Load configured brokers
  useEffect(() => {
    fetchBrokers();
    fetchAlpacaAccount();
  }, []);

  const fetchBrokers = async () => {
    try {
      const res = await fetch(`${API_BASE}/broker/credentials`);
      const data = await res.json();
      setBrokers(data.brokers || []);
    } catch (err) {
      console.error("Failed to fetch brokers:", err);
    }
  };

  const fetchAlpacaAccount = async () => {
    try {
      const res = await fetch(`${API_BASE}/broker/alpaca/account`);
      if (res.ok) {
        const data = await res.json();
        setAlpacaAccount(data);
      }
    } catch (err) {
      // Alpaca not configured yet
      setAlpacaAccount(null);
    }
  };

  const handleSaveCredentials = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setMessage("");

    try {
      const res = await fetch(`${API_BASE}/broker/alpaca/set-creds`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          broker_type: "alpaca",
          api_key: apiKey,
          api_secret: secretKey,
          paper_trading: paperTrading,
        }),
      });

      const data = await res.json();

      if (data.success) {
        setMessage("✅ Credentials saved successfully!");
        setShowCredForm(false);
        setApiKey("");
        setSecretKey("");
        fetchBrokers();
        fetchAlpacaAccount();
      } else {
        setMessage("❌ Failed to save credentials");
      }
    } catch (err) {
      setMessage(`❌ Error: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  const alpacaBroker = brokers.find((b) => b.broker_type === "alpaca");

  return (
    <div className="bg-white border rounded-xl shadow-sm p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold">Broker Configuration</h3>
        <button
          onClick={() => setShowCredForm(!showCredForm)}
          className="px-3 py-1 text-xs bg-black text-white rounded-lg hover:bg-gray-800"
        >
          {showCredForm ? "Cancel" : "Add Broker"}
        </button>
      </div>

      {message && (
        <div className="mb-3 p-2 rounded-lg bg-blue-50 border border-blue-200 text-xs">
          {message}
        </div>
      )}

      {/* Credentials Form */}
      {showCredForm && (
        <form onSubmit={handleSaveCredentials} className="mb-4 p-3 bg-gray-50 rounded-lg border">
          <h4 className="text-xs font-semibold mb-2">Alpaca Credentials</h4>
          <div className="space-y-2">
            <div>
              <label className="text-xs text-gray-600">API Key</label>
              <input
                type="text"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                className="w-full px-2 py-1 text-xs border rounded"
                placeholder="PK..."
                required
              />
            </div>
            <div>
              <label className="text-xs text-gray-600">Secret Key</label>
              <input
                type="password"
                value={secretKey}
                onChange={(e) => setSecretKey(e.target.value)}
                className="w-full px-2 py-1 text-xs border rounded"
                placeholder="Enter secret key"
                required
              />
            </div>
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="paper"
                checked={paperTrading}
                onChange={(e) => setPaperTrading(e.target.checked)}
                className="rounded"
              />
              <label htmlFor="paper" className="text-xs text-gray-600">
                Paper Trading (recommended for testing)
              </label>
            </div>
            <button
              type="submit"
              disabled={loading}
              className="w-full px-3 py-2 text-xs bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
            >
              {loading ? "Saving..." : "Save Credentials"}
            </button>
          </div>
          <p className="text-[10px] text-gray-500 mt-2">
            Get API keys from{" "}
            <a
              href="https://alpaca.markets"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 underline"
            >
              alpaca.markets
            </a>
          </p>
        </form>
      )}

      {/* Configured Brokers */}
      <div className="space-y-3">
        {brokers.length === 0 && !showCredForm && (
          <div className="text-xs text-gray-500 text-center py-4">
            No brokers configured. Click "Add Broker" to get started.
          </div>
        )}

        {alpacaBroker && (
          <div className="p-3 rounded-lg border bg-gray-50">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span className="text-sm font-semibold">Alpaca</span>
                {alpacaBroker.paper_trading && (
                  <span className="px-2 py-0.5 bg-yellow-100 text-yellow-800 text-[10px] rounded">
                    PAPER
                  </span>
                )}
              </div>
              <span className="text-[10px] text-gray-500">
                {alpacaBroker.api_key_prefix}
              </span>
            </div>

            {alpacaAccount && (
              <div className="grid grid-cols-2 gap-2 mt-2">
                <div className="p-2 bg-white rounded border">
                  <div className="text-[10px] text-gray-500">Portfolio Value</div>
                  <div className="text-sm font-semibold">
                    ${alpacaAccount.portfolio_value.toFixed(2)}
                  </div>
                </div>
                <div className="p-2 bg-white rounded border">
                  <div className="text-[10px] text-gray-500">Cash</div>
                  <div className="text-sm font-semibold">
                    ${alpacaAccount.cash.toFixed(2)}
                  </div>
                </div>
                <div className="p-2 bg-white rounded border">
                  <div className="text-[10px] text-gray-500">Buying Power</div>
                  <div className="text-sm font-semibold">
                    ${alpacaAccount.buying_power.toFixed(2)}
                  </div>
                </div>
                <div className="p-2 bg-white rounded border">
                  <div className="text-[10px] text-gray-500">Equity</div>
                  <div className="text-sm font-semibold">
                    ${alpacaAccount.equity.toFixed(2)}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
