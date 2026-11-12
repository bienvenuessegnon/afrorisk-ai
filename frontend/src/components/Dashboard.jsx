import Factors from "./Factors";
import { MODEL_LABELS, formatDecimal, formatPercent } from "../labels";

function Cards({ metrics }) {
  const items = [
    ["ROC-AUC", formatDecimal(metrics.roc_auc)],
    ["PR-AUC", formatDecimal(metrics.pr_auc)],
    ["Précision", formatDecimal(metrics.precision)],
    ["Rappel", formatDecimal(metrics.recall)],
    ["F1", formatDecimal(metrics.f1)],
  ];
  return (
    <div className="metric-grid">
      {items.map(([label, value]) => (
        <article key={label} className="metric">
          <p>{label}</p>
          <strong>{value}</strong>
        </article>
      ))}
    </div>
  );
}

function Comparison({ rows }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Modèle</th>
            <th>ROC-AUC</th>
            <th>PR-AUC</th>
            <th>Précision</th>
            <th>Rappel</th>
            <th>F1</th>
            <th>Seuil</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.model}>
              <td>{MODEL_LABELS[row.model] || row.model}</td>
              <td>{formatDecimal(row.roc_auc)}</td>
              <td>{formatDecimal(row.pr_auc)}</td>
              <td>{formatDecimal(row.precision)}</td>
              <td>{formatDecimal(row.recall)}</td>
              <td>{formatDecimal(row.f1)}</td>
              <td>{formatDecimal(row.threshold)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Dashboard({ metrics, error }) {
  if (error) {
    return <p className="banner error">{error}</p>;
  }
  if (!metrics) {
    return <p className="muted">Chargement des indicateurs…</p>;
  }

  const served = metrics.served;
  const matrix = served.metrics.confusion_matrix || [];
  const sourceLabel = metrics.placeholder
    ? "Fixture synthétique — ces indicateurs ne décrivent pas Home Credit."
    : "Home Credit — indicateurs pédagogiques, pas une décision d'octroi.";

  return (
    <div className="stack">
      <p className={metrics.placeholder ? "banner warn" : "banner"}>{sourceLabel}</p>
      <section>
        <div className="section-title">
          <h2>Modèle servi</h2>
          <p>
            {MODEL_LABELS[served.selected_model] || served.selected_model} · seuil{" "}
            {formatDecimal(served.threshold)} · {metrics.cv_folds} plis stratifiés ·{" "}
            {metrics.n_rows.toLocaleString("fr-FR")} demandes · défaut{" "}
            {formatPercent(metrics.positive_rate)}
          </p>
        </div>
        <Cards metrics={served.metrics} />
      </section>

      <section className="split">
        <article className="card">
          <h3>Matrice de confusion</h3>
          <p className="muted">Prédictions hors pli, au seuil de coût. Lignes : réel. Colonnes : prédit.</p>
          <table className="matrix">
            <thead>
              <tr>
                <th></th>
                <th>Prédit non-défaut</th>
                <th>Prédit défaut</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th>Réel non-défaut</th>
                <td>{matrix[0]?.[0] ?? "—"}</td>
                <td>{matrix[0]?.[1] ?? "—"}</td>
              </tr>
              <tr>
                <th>Réel défaut</th>
                <td>{matrix[1]?.[0] ?? "—"}</td>
                <td>{matrix[1]?.[1] ?? "—"}</td>
              </tr>
            </tbody>
          </table>
        </article>
        <article className="card">
          <h3>Facteurs globaux</h3>
          <p className="muted">Importance moyenne absolue SHAP sur un échantillon du jeu d'entraînement.</p>
          <Factors
            signed={false}
            factors={(served.global_factors || []).slice(0, 6).map((item) => ({
              ...item,
              contribution: item.mean_abs_shap,
            }))}
          />
        </article>
      </section>

      <section className="card">
        <h3>Comparaison — variables du formulaire</h3>
        <Comparison rows={served.comparison} />
      </section>
      <section className="card">
        <h3>Comparaison — modèle de recherche</h3>
        <p className="muted">
          Application et agrégations des tables bureau, historique bureau, demandes précédentes, TPE,
          échéances et cartes. Modèle retenu :{" "}
          {MODEL_LABELS[metrics.research.selected_model] || metrics.research.selected_model}.
        </p>
        <Comparison rows={metrics.research.comparison} />
      </section>
    </div>
  );
}
