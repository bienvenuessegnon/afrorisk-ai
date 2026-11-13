export default function Factors({ factors, signed = true }) {
  if (!factors?.length) {
    return <p className="muted">Aucun facteur à afficher.</p>;
  }
  const max = Math.max(...factors.map((item) => Math.abs(item.contribution)), 0.0001);
  return (
    <ul className="factors">
      {factors.map((item) => {
        const width = `${(Math.abs(item.contribution) / max) * 100}%`;
        const tone = signed ? (item.contribution > 0 ? "up" : "down") : "neutral";
        return (
          <li key={item.feature}>
            <div className="factor-head">
              <span>{item.label}</span>
              {signed ? (
                <span className={`direction ${tone}`}>
                  {item.direction === "augmente" ? "Augmente le risque" : "Réduit le risque"}
                </span>
              ) : (
                <span className="muted">{item.contribution.toLocaleString("fr-FR", { maximumFractionDigits: 3 })}</span>
              )}
            </div>
            <div className="track" aria-hidden="true">
              <div className={`bar ${tone}`} style={{ width }} />
            </div>
          </li>
        );
      })}
    </ul>
  );
}
