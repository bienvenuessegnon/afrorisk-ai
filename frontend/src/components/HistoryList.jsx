import { formatPercent } from "../labels";

const RISK = { faible: "Faible", modéré: "Modéré", élevé: "Élevé" };

export default function HistoryList({ items, onOpen, onRemove, onClear }) {
  if (!items.length) {
    return (
      <section className="card">
        <h2>Historique</h2>
        <p className="muted">Les simulations de ce navigateur seront conservées ici.</p>
      </section>
    );
  }

  return (
    <section className="card">
      <div className="form-head">
        <h2>Historique</h2>
        <button type="button" className="ghost" onClick={onClear}>
          Tout effacer
        </button>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Date</th>
              <th>Probabilité</th>
              <th>Niveau</th>
              <th>Crédit</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {items.map((item) => (
              <tr key={item.id}>
                <td>{new Date(item.at).toLocaleString("fr-FR")}</td>
                <td>{formatPercent(item.result.probability_default)}</td>
                <td>{RISK[item.result.risk_level] || item.result.risk_level}</td>
                <td>{Number(item.form.amt_credit).toLocaleString("fr-FR")}</td>
                <td className="row-actions">
                  <button type="button" className="ghost" onClick={() => onOpen(item)}>
                    Revoir
                  </button>
                  <button type="button" className="ghost" onClick={() => onRemove(item.id)}>
                    Retirer
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
