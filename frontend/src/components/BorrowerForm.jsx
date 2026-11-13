import { CONTRACTS, EDUCATION, EMPTY_FORM, EXAMPLE_FORM, FAMILY, INCOME } from "../labels";

function Field({ label, hint, children }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
      {hint ? <small>{hint}</small> : null}
    </label>
  );
}

function Select({ value, onChange, options }) {
  return (
    <select value={value} onChange={(event) => onChange(event.target.value)}>
      {options.map(([optionValue, optionLabel]) => (
        <option key={optionValue} value={optionValue}>
          {optionLabel}
        </option>
      ))}
    </select>
  );
}

export default function BorrowerForm({ form, onChange, onSubmit, pending }) {
  const set = (key) => (value) => onChange({ ...form, [key]: value });
  const input = (key, props = {}) => (
    <input
      value={form[key]}
      onChange={(event) => set(key)(event.target.value)}
      {...props}
    />
  );

  return (
    <form
      className="card form"
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit();
      }}
    >
      <div className="form-head">
        <h2>Dossier</h2>
        <button type="button" className="ghost" onClick={() => onChange({ ...EXAMPLE_FORM })}>
          Remplir un exemple
        </button>
      </div>
      <div className="form-grid">
        <Field label="Revenu annuel" hint="Montant annuel, pas mensuel.">
          {input("amt_income_total", { type: "number", min: "1", required: true })}
        </Field>
        <Field label="Montant du crédit">{input("amt_credit", { type: "number", min: "1", required: true })}</Field>
        <Field label="Annuité">{input("amt_annuity", { type: "number", min: "1", step: "0.01", required: true })}</Field>
        <Field label="Prix du bien">{input("amt_goods_price", { type: "number", min: "1", required: true })}</Field>
        <Field label="Âge (années)">{input("age_years", { type: "number", min: "18", max: "100", required: true })}</Field>
        <Field label="Ancienneté professionnelle (années)" hint="Vide si non renseignée.">
          {input("employment_years", { type: "number", min: "0", max: "70", step: "0.1" })}
        </Field>
        <Field label="Score externe 1" hint="Entre 0 et 1, facultatif.">
          {input("ext_source_1", { type: "number", min: "0", max: "1", step: "0.01" })}
        </Field>
        <Field label="Score externe 2">{input("ext_source_2", { type: "number", min: "0", max: "1", step: "0.01" })}</Field>
        <Field label="Score externe 3">{input("ext_source_3", { type: "number", min: "0", max: "1", step: "0.01" })}</Field>
        <Field label="Type de contrat">
          <Select value={form.name_contract_type} onChange={set("name_contract_type")} options={CONTRACTS} />
        </Field>
        <Field label="Niveau d'études">
          <Select value={form.name_education_type} onChange={set("name_education_type")} options={EDUCATION} />
        </Field>
        <Field label="Situation familiale">
          <Select value={form.name_family_status} onChange={set("name_family_status")} options={FAMILY} />
        </Field>
        <Field label="Type de revenu">
          <Select value={form.name_income_type} onChange={set("name_income_type")} options={INCOME} />
        </Field>
        <Field label="Nombre d'enfants">{input("cnt_children", { type: "number", min: "0", max: "20", required: true })}</Field>
        <Field label="Propriétaire immobilier">
          <Select
            value={form.flag_own_realty}
            onChange={set("flag_own_realty")}
            options={[
              ["Y", "Oui"],
              ["N", "Non"],
            ]}
          />
        </Field>
        <Field label="Propriétaire d'un véhicule">
          <Select
            value={form.flag_own_car}
            onChange={set("flag_own_car")}
            options={[
              ["Y", "Oui"],
              ["N", "Non"],
            ]}
          />
        </Field>
        <Field label="Note de la région" hint="Échelle Home Credit : 1 meilleure note, 3 plus faible.">
          <Select
            value={form.region_rating_client}
            onChange={set("region_rating_client")}
            options={[
              ["1", "1"],
              ["2", "2"],
              ["3", "3"],
            ]}
          />
        </Field>
      </div>
      <div className="form-actions">
        <button type="submit" disabled={pending}>
          {pending ? "Calcul…" : "Estimer le risque"}
        </button>
        <button type="button" className="ghost" onClick={() => onChange({ ...EMPTY_FORM })}>
          Effacer
        </button>
      </div>
    </form>
  );
}
