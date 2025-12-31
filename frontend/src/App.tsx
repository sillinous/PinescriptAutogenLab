import { useState } from "react";
import PineLabUnifiedDashboard from "./dashboard/PineLabUnifiedDashboard";
import BrokerPanel from "./dashboard/BrokerPanel";

export default function App() {
  const [activeTab, setActiveTab] = useState<"dashboard" | "broker">("dashboard");

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            <h1 className="text-lg font-bold">PineScript Autogen Lab</h1>
            <div className="flex gap-2">
              <button
                onClick={() => setActiveTab("dashboard")}
                className={`px-4 py-2 text-sm rounded-lg transition-colors ${
                  activeTab === "dashboard"
                    ? "bg-black text-white"
                    : "bg-gray-100 hover:bg-gray-200"
                }`}
              >
                Dashboard
              </button>
              <button
                onClick={() => setActiveTab("broker")}
                className={`px-4 py-2 text-sm rounded-lg transition-colors ${
                  activeTab === "broker"
                    ? "bg-black text-white"
                    : "bg-gray-100 hover:bg-gray-200"
                }`}
              >
                Brokers
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-4 py-6">
        {activeTab === "dashboard" && <PineLabUnifiedDashboard />}
        {activeTab === "broker" && <BrokerPanel />}
      </div>
    </div>
  );
}
