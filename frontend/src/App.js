import { useEffect } from "react";
import "@/App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import axios from "axios";
import { useLocation, useNavigate } from "react-router-dom";
import { HOME, LOGIN, REGISTER, LOGOUT } from "@/constants/testIds";
import { useTheme } from "@radix-ui/react-theming";
import { Toaster, useToast } from "@/components/ui/sonner";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:5555";
const API = `${BACKEND_URL}/api`;

const testIds = {
  ...HOME,
  ...LOGIN,
  ...REGISTER,
  ...LOGOUT,
};

const DashboardLayout = ({
  children,
}: {
  children: React.ReactNode;
}) => {
  return (
    <div className="dashboard-layout">
      <Sidebar />
      <main className="dashboard-main">
        <Header />
        <section className="dashboard-content">
          {children}
        </section>
      </main>
    </div>
  );
};

const Sidebar = () => {
  const navigate = useNavigate();
  const [theme] = useTheme();

  return (
    <aside className="sidebar collapsed-until-md">
      <div className="sidebar-header">
        <div className="sidebar-brand">
          <span className="icon">🧠</span>
          <span className="brand-text">Grim Hybrid</span>
        </div>
      </div>

      <nav>
        <ul className="nav-links">
          <li>
            <a
              href="/"
              className={navigate() === "/" ? "active" : ""}
              data-testid="sidebar-home-link"
            >
              <span className="nav-icon">🏠</span>
              <span className="nav-label">Home</span>
            </a>
          </li>
          <li>
            <a
              href="/signals"
              className={navigate() === "/signals" ? "active" : ""}
              data-testid="sidebar-signals-link"
            >
              <span className="nav-icon">📡</span>
              <span className="nav-label">Signals</span>
            </a>
          </li>
          <li>
            <a
              href "/models"
              className={navigate() === "/models" ? "active" : ""}
              data-testid="sidebar-models-link"
            >
              <span className="nav-icon">🤖</span>
              <span className="nav-label">Models</span>
            </a>
          </li>
          <li>
            <a
              href "/risk"
              className={navigate() === "/risk" ? "active" : ""}
              data-testid="sidebar-risk-link"
            >
              <span className="nav-icon">⚠️</span>
              <span className="nav-label">Risk</span>
            </a>
          </li>
          <li>
            <a
              "/settings"
              className={navigate() === "/settings" ? "active" : ""}
              data-testid="sidebar-settings-link"
            >
              <span className="nav-icon">⚙️</span>
              <span className="nav-label">Settings</span>
            </a>
          </li>
        </ul>
      </nav>

      <div className="sidebar-footer">
        <button
          onClick={() => navigate("/login")}
          className="btn-link text-sm"
          data-testid="sidebar-logout-btn"
        >
          Logout
        </button>
      </div>
    </aside>
  );
};

const Header = () => {
  const navigate = useNavigate();
  const { toaster } = useToast();

  return (
    <header className="dashboard-header">
      <div className="header-left">
        <button
          className="menu-toggle"
          onClick={() => navigate("/")}
          aria-label="Menu"
          data-testid="header-menu-btn"
        >
          ☰
        </button>
        <div className="header-brand">
          <span className="brand-icon">🧠</span>
          <span className="brand-title">Grim Hybrid</span>
        </div>
      </div>

      <div className="header-right">
        <div className="user-profile">
          <div className="user-avatar">
            <span className="avatar-initials">CH</span>
          </div>
          <button
            className="user-dropdown-toggle"
            aria-label="User menu"
            data-testid="user-dropdown-toggle"
          >
            <span className="user-name">User</span>
            <svg className="arrow-icon" viewBox="0 0 24 24">
              <path fill="currentColor" d="M7 10l5 5 5-5z" />
            </svg>
          </button>
          <ul className="user-dropdown-menu" role="menu">
            <li>
              <a
                href "/login"
                className="dropdown-item"
                data-testid="dropdown-login"
              >
                Login
              </a>
            </li>
            <li>
              <a
                href "/register"
                className="dropdown-item"
                data-testid="dropdown-register"
              >
                Register
              </a>
            </li>
            <li>
              <hr className="dropdown-divider" />
            </li>
            <li>
              <a
                href "javascript:void(0)"
                className="dropdown-item"
                onClick={() => navigate("/login")}
                data-testid="dropdown-logout"
              >
                Logout
              </a>
            </li>
          </ul>
        </div>
      </div>
    </header>
  );
};

const HomePage = () => {
  const navigate = useNavigate();
  const [theme] = useTheme();
  const { toaster } = useToast();

  useEffect(() => {
    // Fetch system status on page load
    const fetchSystemStatus = async () => {
      try {
        const response = await axios.get(`${API}/`);
        console.log("System status:", response.data);
      } catch (e) {
        console.error("Failed to fetch system status:", e);
        toaster.error("Failed to connect to trading system");
      }
    };

    fetchSystemStatus();

    // Set up periodic status refresh
    const interval = setInterval(fetchSystemStatus, 5000);
    return () => clearInterval(interval);
  }, [navigate, toaster]);

  return (
    <div className="home-page">
      <header className="home-header">
        <h1 className="heading-xl">
          <span className="gradient-text">Grim Hybrid</span>
          <span className="subheading">AI Trading System Dashboard</span>
        </h1>
        <p className="home-desc">
          Modular AI trading framework with real-time market analysis and
          intelligent trade execution
        </p>
      </header>

      <main className="home-main">
        <div className="stats-grid">
          <StatCard
            title="System Status"
            value="Connecting..."
            icon="🟢"
            testid="stat-system-status"
          />
          <StatCard
            title="Active Models"
            value="0"
            icon="🤖"
            testid="stat-models-count"
          />
          <StatCard
            title "Daily P&L"
            value="$0.00"
            icon="💰"
            testid="stat-daily-pnl"
          />
          <StatCard
            title="Total Trades"
            value="0"
            icon="📊"
            testid="stat-total-trades"
          />
        </div>

        <div className="charts-grid">
          <ChartCard
            title="Recent Signals"
            testid="chart-recent-signals"
            type="area"
          />
          <ChartCard
            title="Score Distribution"
            testid="chart-score-distribution"
            type="bar"
          />
          <ChartCard
            title="Session Activity"
            testid="chart-session-activity"
            type="line"
          />
        </div>
      </main>
    </div>
  );
};

const StatCard = ({
  title,
  value,
  icon,
  testid,
}: {
  title: string;
  value: string;
  icon: string;
  testid: string;
}) => {
  return (
    <div className="stat-card" data-testid={testid}>
      <div className="stat-icon">
        {icon}
      </div>
      <div className="stat-value">{value}</div>
      <div className="stat-title">{title}</div>
    </div>
  );
};

const ChartCard = ({
  title,
  testid,
  type,
}: {
  title: string;
  testid: string;
  type: string;
}) => {
  return (
    <div className="chart-card" data-testid={testid}>
      <h3 className="chart-title">{title}</h3>
      <div className="chart-placeholder">
        <svg
          width="400"
          height="250"
          viewBox="0 0 400 250"
          className="chart-svg"
        >
          <rect width="400" height="250" fill="var(--chart-bg, #1a1a2e)" />
          <text
            x="200"
            y="125"
            textAnchor="middle"
            fill="var(--text-secondary, #888)"
            fontSize="14"
          >
            {type} Chart
          </text>
        </svg>
      </div>
    </div>
  );
};

const SignalsPage = () => {
  const { toaster } = useToast();
  const [signals, setSignals] = React.useState([]);

  useEffect(() => {
    const fetchSignals = async () => {
      try {
        const response = await axios.get(`${API}/signals`, {
          timeout: 10000,
        });
        setSignals(response.data || []);
      } catch (e) {
        console.error("Failed to fetch signals:", e);
        toaster.error("Failed to load signals data");
      }
    };

    fetchSignals();

    const interval = setInterval(fetchSignals, 3000);
    return () => clearInterval(interval);
  }, [toaster]);

  return (
    <div className="signals-page">
      <header className="page-header">
        <h2 className="heading-lg">Recent Trading Signals</h2>
        <button
          onClick={() => navigate("/")}
          className="btn-icon btn-secondary"
          aria-label="Back to home"
          data-testid="back-home-btn"
        >
          ←
        </button>
      </header>

      {signals.length === 0 ? (
        <div className="empty-state">
          <svg
            width="64"
            height="64"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            className="empty-icon"
          >
            <circle cx="9" cy="21" r="1" />
            <circle cx="20" cy="21" r="1" />
            <path d="M1 1l4 4l12 12l4-4M1 7l4 4l12 12l4-4" />
          </svg>
          <p>No recent signals</p>
        </div>
      ) : (
        <div className="signals-list">
          {signals.map((signal, index) => (
            <SignalCard
              key={index}
              signal={signal}
              testid={`signal-item-${signal.symbol || index}`}
            />
          ))}
        </div>
      )}
    </div>
  );
};

const SignalCard = ({
  signal,
  testid,
}: {
  signal: any;
  testid: string;
}) => {
  const action = signal.action || "HOLD";
  const confidence = signal.confidence || 0;
  const symbol = signal.symbol || "UNKNOWN";
  const setupScore = signal.setup_score || 0;

  return (
    <div
      className="signal-card"
      data-testid={testid}
      style={{ borderLeft: `4px solid ${getActionColor(action)}` }}
    >
      <div className="signal-header">
        <span className="signal-symbol">{symbol}</span>
        <span className="signal-action">{action}</span>
      </div>

      <div className="signal-details">
        <div className="detail-row">
          <span className="detail-label">Confidence</span>
          <span className="detail-value">{confidence.toFixed(1)}%</span>
        </div>
        <div className="detail-row">
          <span className="detail-label">Setup Score</span>
          <span className="detail-value">{setupScore.toFixed(2)}</span>
        </div>
        <div className="detail-row">
          <span className="detail-label">Method</span>
          <span className="detail-value">{signal.method || "N/A"}</span>
        </div>
        <div className="detail-row">
          <span className="detail-label">Model</span>
          <span className="detail-value">{signal.model_type || "N/A"}</span>
        </div>
      </div>

      <div className="signal-risk">
        <div className="risk-item">
          <span className="risk-label">Risk %</span>
          <span className="risk-value">{signal.risk_pct?.toFixed(2) || "0.00"}%</span>
        </div>
        <div className="risk-item">
          <span className="risk-label">SL Dist</span>
          <span className="risk-value">{signal.sl_distance || "0"} pips</span>
        </div>
      </div>
    </div>
  );
};

const ModelsPage = () => {
  const { toaster } = useToast();
  const [modelStatus, setModelStatus] = React.useState([]);

  useEffect(() => {
    const fetchModelStatus = async () => {
      try {
        const response = await axios.get(`${API}/models/status`, {
          timeout: 10000,
        });
        setModelStatus(response.data || []);
      } catch (e) {
        console.error("Failed to fetch model status:", e);
        toaster.error("Failed to load model status");
      }
    };

    fetchModelStatus();

    const interval = setInterval(fetchModelStatus, 5000);
    return () => clearInterval(interval);
  }, [toaster]);

  return (
    <div className="models-page">
      <header className="page-header">
        <h2 className="heading-lg">Model Registry</h2>
        <p className="page-subtitle">Monitor model availability and version status</p>
        <button
          onClick={() => navigate("/")}
          className="btn-icon btn-secondary"
          aria-label="Back to home"
          data-testid="back-home-models"
        >
          ←
        </button>
      </header>

      {modelStatus.length === 0 ? (
        <div className="empty-state">
          <svg
            width="64"
            height="64"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            className="empty-icon"
          >
            <circle cx="9" cy="21" r="1" />
            <circle cx="20" cy="21" r="1" />
            <path d="M1 1l4 4l12 12l4-4M1 7l4 4l12 12l4-4" />
          </svg>
          <p>No model data available</p>
        </div>
      ) : (
        <div className="models-table">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Status</th>
                <th>SKLearn</th>
                <th>XGBoost</th>
                <th>Version Check</th>
              </tr>
            </thead>
            <tbody>
              {modelStatus.map((row, index) => (
                <tr key={index} data-testid={`model-row-${row.name}`}>
                  <td>{row.name}</td>
                  <td>
                    <span
                      className={
                        row.status === "REACHED"
                          ? "status-reached"
                          : row.status === "MISSING"
                          ? "status-missing"
                          : row.status === "FAILED"
                          ? "status-failed"
                          : ""
                      }
                    >
                      {row.status}
                    </span>
                  </td>
                  <td>{row.trained_sklearn || "N/A"}</td>
                  <td>{row.trained_xgboost || "N/A"}</td>
                  <td>
                    <span
                      className={
                        row.version_check === "MATCH"
                          ? "status-match"
                          : row.version_check === "MISMATCH"
                          ? "status-mismatch"
                          : "status-unknown"
                      }
                    >
                      {row.version_check}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="models-summary">
            <span>Models reached: {modelStatus.filter(
              (r) => r.status === "REACHED"
            ).length}/{modelStatus.length}</span>
            {modelStatus.some((r) => r.version_check === "MISMATCH") && (
              <span className="warning">
                ⚠️ Version mismatch detected - predictions may be invalid
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

const RiskPage = () => {
  const { toaster } = useToast();
  const [riskState, setRiskState] = React.useState(null);

  useEffect(() => {
    const fetchRiskState = async () => {
      try {
        const response = await axios.get(`${API}/risk/state`, {
          timeout: 10000,
        });
        setRiskState(response.data || null);
      } catch (e) {
        console.error("Failed to fetch risk state:", e);
        toaster.error("Failed to load risk state");
      }
    };

    fetchRiskState();

    const interval = setInterval(fetchRiskState, 2000);
    return () => clearInterval(interval);
  }, [toaster]);

  if (!riskState) {
    return (
      <div className="risk-page">
        <header className="page-header">
          <h2 className="heading-lg">Risk Engine</h2>
          <p className="page-subtitle">Daily risk metrics and exposure</p>
        </header>

        <div className="risk-overview">
          <div className="stat-card pending">
            <div className="stat-icon">⚠️</div>
            <div className="stat-value">Loading...</div>
            <div className="stat-title">Daily P&L</div>
          </div>
          <div className="stat-card pending">
            <div className="stat-icon">📊</div>
            <div className="stat-value">0</div>
            <div className="stat-title">Daily Trades</div>
          </div>
          <div className="stat-card pending">
            <div className="stat-icon">💵</div>
            <div className="stat-value">0.00%</div>
            <div className="stat-title">Max Daily Loss</div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="risk-page">
      <header className="page-header">
        <h2 className="heading-lg">Risk Engine</h2>
        <p className="page-subtitle">Daily risk metrics and exposure</p>
        <button
          onClick={() => navigate("/")}
          className="btn-icon btn-secondary"
          aria-label="Back to home"
          data-testid="back-home-risk"
        >
          ←
        </button>
      </header>

      <div className="risk-details">
        <div className="daily-exposure">
          <h3>Daily Exposure</h3>
          <div className.exposure-bar>
            <div
              className="exposure-fill"
              style={{
                width: `${riskState.daily_pnl_pct / 4 * 100}%`,
                backgroundColor:
                  riskState.daily_pnl_pct < 0
                    ? "var(--danger, #ef4444)"
                    : riskState.daily_pnl_pct < 2
                    ? "var(--warning, #f59e0b)"
                    : "var(--success, #10b981)",
              }}
            />
          </div>
          <span className="exposure-value">
            {riskState.daily_pnl_pct.toFixed(2)}% / {riskState.max_daily_loss_pct.toFixed(
              2
            )}%
          </span>
        </div>

        <div className="trade-counts">
          <div className="trade-item">
            <span>Daily Trades:</span>
            <span className="trade-value">
              {riskState.daily_trades}/{riskState.max_daily_trades}
            </span>
          </div>
          <div className="trade-item">
            <span>Open Trades:</span>
            <span className="trade-value">{riskState.open_trades}</span>
          </div>
        </div>

        <div className="risk-caps">
          <h3>Risk Configuration</h3>
          <div className="cap-item">
            <span>Max Concurrent Trades:</span>
            <span>{riskState.max_concurrent_trades}</span>
          </div>
          <div className="cap-item">
            <span>Max Daily Trades:</span>
            <span>{riskState.max_daily_trades}</span>
          </div>
          <div className="cap-item">
            <span>Max Daily Loss %:</span>
            <span>{riskState.max_daily_loss_pct}%</span>
          </div>
          <div className="cap-item">
            <span>Base Risk %:</span>
            <span>{riskState.base_risk_pct}%</span>
          </div>
        </div>
      </div>
    </div>
  );
};

const SettingsPage = () => {
  const { toaster } = useToast();

  return (
    <div className="settings-page">
      <header className="page-header">
        <h2 className="heading-lg">Settings</h2>
        <p className="page-subtitle">System configuration</p>
        <button
          onClick={() => navigate("/")}
          className="btn-icon btn-secondary"
          aria-label="Back to home"
          data-testid="back-home-settings"
        >
          ←
        </button>
      </header>

      <div className="settings-content">
        <div className="setting-group">
          <h3>Backend Connection</h3>
          <div className="setting-row">
            <span>Backend URL:</span>
            <input
              type="text"
              value="http://localhost:5555"
              readOnly
              className="setting-input"
              data-testid="setting-backend-url"
            />
          </div>
          <button
            onClick=testConnection
            className="btn-sm"
            data-testid="test-connection-btn"
          >
            Test Connection
          </button>
        </div>

        <div className="setting-group">
          <h3>Dashboard Refresh</h3>
          <div className="setting-row">
            <span>Auto-refresh interval (seconds):</span>
            <input
              type="number"
              value "30"
              min="5"
              max="300"
              className="setting-input"
              data-testid="refresh-interval"
            />
          </div>
        </div>

        <div className="setting-group">
          <h3>Display Preferences</h3>
          <div className="setting-row">
            <span>Theme:</span>
            <select
              className="setting-select"
              data-testid="theme-select"
            >
              <option value="light">Light</option>
              <option value="dark">Dark</option>
            </select>
          </div>
        </div>

        <button
          onClick={() => toaster.info("Settings saved successfully")}
          className="btn-primary"
          data-testid="save-settings-btn"
        >
          Save Settings
        </button>
      </div>
    </div>
  );
};

const getActionColor = (action: string) => {
  const colors = {
    BUY: "var(--success, #10b981)",
    SELL: "var(--warning, #f59e0b)",
    HOLD: "var(--muted, #6b7280)",
  };
  return colors[action] || colors.HOLD;
};

export default function App() {
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <div className="app-root">
      <DashboardLayout>

        {/* Flash message for toast notifications */}
        {location.state?.toast && (
          <div
            className="toast-notification"
            data-testid="flash-toast"
          >
            {location.state.toast.message}
          </div>
        )}

        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/signals" element={<SignalsPage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/risk" element={<RiskPage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Routes>
      </DashboardLayout>
    </div>
  );
}