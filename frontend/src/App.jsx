import { useEffect, useState } from "react";
import { fetchMetrics, predict } from "./api";
import BorrowerForm from "./components/BorrowerForm";
import Dashboard from "./components/Dashboard";
import HistoryList from "./components/HistoryList";
import ResultPanel from "./components/ResultPanel";
import { EMPTY_FORM, toPayload } from "./labels";

const STORAGE_KEY = "afrorisk.history";

function readHistory() {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export default function App() {
  const [tab, setTab] = useState("analyse");
  const [metrics, setMetrics] = useState(null);
  const [metricsError, setMetricsError] = useState("");
  const [form, setForm] = useState(EMPTY_FORM);
  const [result, setResult] = useState(null);
  const [formError, setFormError] = useState("");
  const [pending, setPending] = useState(false);
  const [history, setHistory] = useState(readHistory);

  useEffect(() => {
    fetchMetrics()
      .then(setMetrics)
      .catch(() => setMetricsError("Les indicateurs ne sont pas disponibles. Vérifiez que l'API est démarrée."));
  }, []);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
  }, [history]);

  async function onSubmit() {
    setPending(true);
    setFormError("");
    try {
      const payload = toPayload(form);
      const next = await predict(payload);
      setResult(next);
      setHistory((current) =>
        [
          {
            id: crypto.randomUUID(),
            at: new Date().toISOString(),
            form,
            result: next,
          },
          ...current,
        ].slice(0, 20),
      );
    } catch (error) {
      setResult(null);
      setFormError(error.message || "Calcul impossible.");
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="page">
      <header className="topbar">
        <div className="brand">
          <p className="mark">AfroRisk</p>
          <p className="tag">Turning financial data into explainable risk intelligence</p>
        </div>
        {metricsError ? <span className="pill élevé">API indisponible</span> : null}
        {metrics?.placeholder ? <span className="pill élevé">Démo synthétique</span> : null}
        {metrics && !metrics.placeholder ? <span className="pill faible">Modèle Home Credit</span> : null}
      </header>

      <div className="shell">
        <nav className="tabs" aria-label="Sections">
          <button type="button" className={tab === "dashboard" ? "active" : ""} onClick={() => setTab("dashboard")}>
            Indicateurs
          </button>
          <button type="button" className={tab === "analyse" ? "active" : ""} onClick={() => setTab("analyse")}>
            Analyse
          </button>
          <button type="button" className={tab === "history" ? "active" : ""} onClick={() => setTab("history")}>
            Historique
          </button>
        </nav>

        {tab === "dashboard" ? <Dashboard metrics={metrics} error={metricsError} /> : null}

        {tab === "analyse" ? (
          <div className="workspace">
            <BorrowerForm form={form} onChange={setForm} onSubmit={onSubmit} pending={pending} />
            <ResultPanel result={result} error={formError} />
          </div>
        ) : null}

        {tab === "history" ? (
          <HistoryList
            items={history}
            onOpen={(item) => {
              setForm(item.form);
              setResult(item.result);
              setFormError("");
              setTab("analyse");
            }}
            onRemove={(id) => setHistory((current) => current.filter((item) => item.id !== id))}
            onClear={() => setHistory([])}
          />
        ) : null}

        <footer>
          Usage pédagogique. Les données Home Credit ne représentent pas le Bénin ni l'Afrique de l'Ouest.
          Le score n'est jamais une décision définitive d'octroi.
        </footer>
      </div>
    </div>
  );
}
