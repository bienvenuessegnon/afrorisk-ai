import Factors from "./Factors";
import Gauge from "./Gauge";

const RISK = {
  faible: "Faible",
  modéré: "Modéré",
  élevé: "Élevé",
};

export default function ResultPanel({ result, error }) {
  if (error) {
    return (
      <aside className="card result">
        <h2>Résultat</h2>
        <p className="banner error">{error}</p>
      </aside>
    );
  }
  if (!result) {
    return (
      <aside className="card result empty">
        <h2>Résultat</h2>
        <p>La probabilité de défaut et les facteurs apparaîtront ici après l'envoi du dossier.</p>
      </aside>
    );
  }

  return (
    <aside className="card result">
      <div className="result-head">
        <h2>Probabilité de défaut</h2>
        <span className={`pill ${result.risk_level}`}>{RISK[result.risk_level] || result.risk_level}</span>
      </div>
      <Gauge value={result.probability_default} />
      <p className="result-class">
        Classe au seuil {result.threshold.toLocaleString("fr-FR")} : <strong>{result.predicted_label}</strong>
      </p>
      <h3>Facteurs de ce dossier</h3>
      <Factors factors={result.factors} />
      {result.model?.placeholder ? (
        <p className="banner warn">Modèle d'illustration entraîné sur des données synthétiques.</p>
      ) : (
        <p className="banner">Estimation d'aide à la lecture. Ce n'est pas une décision d'octroi.</p>
      )}
    </aside>
  );
}
