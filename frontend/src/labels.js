export const MODEL_LABELS = {
  logistic_regression: "Régression logistique",
  decision_tree: "Arbre de décision",
  random_forest: "Forêt aléatoire",
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
};

export const CONTRACTS = [
  ["Cash loans", "Prêt de trésorerie"],
  ["Revolving loans", "Crédit renouvelable"],
];

export const EDUCATION = [
  ["Secondary / secondary special", "Secondaire"],
  ["Higher education", "Enseignement supérieur"],
  ["Incomplete higher", "Supérieur incomplet"],
  ["Lower secondary", "Premier cycle du secondaire"],
  ["Academic degree", "Diplôme universitaire"],
];

export const FAMILY = [
  ["Married", "Marié"],
  ["Civil marriage", "Union libre"],
  ["Single / not married", "Célibataire"],
  ["Separated", "Séparé"],
  ["Widow", "Veuf ou veuve"],
];

export const INCOME = [
  ["Working", "Salarié"],
  ["Commercial associate", "Associé commercial"],
  ["State servant", "Agent public"],
  ["Pensioner", "Retraité"],
  ["Unemployed", "Sans emploi"],
];

export const EMPTY_FORM = {
  amt_income_total: "",
  amt_credit: "",
  amt_annuity: "",
  amt_goods_price: "",
  age_years: "",
  employment_years: "",
  ext_source_1: "",
  ext_source_2: "",
  ext_source_3: "",
  name_contract_type: "Cash loans",
  name_education_type: "Secondary / secondary special",
  name_family_status: "Married",
  name_income_type: "Working",
  cnt_children: "0",
  flag_own_realty: "Y",
  flag_own_car: "N",
  region_rating_client: "2",
  code_gender: "",
};

export const EXAMPLE_FORM = {
  ...EMPTY_FORM,
  amt_income_total: "145000",
  amt_credit: "480000",
  amt_annuity: "26500",
  amt_goods_price: "420000",
  age_years: "38",
  employment_years: "7",
  ext_source_1: "0.42",
  ext_source_2: "0.58",
  ext_source_3: "0.36",
  name_education_type: "Higher education",
  cnt_children: "1",
};

const number = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 0 });
const decimal = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 3 });

export function formatAmount(value) {
  return number.format(value);
}

export function formatDecimal(value) {
  if (value === null || value === undefined || value === "") return "—";
  return decimal.format(value);
}

export function formatPercent(value) {
  return new Intl.NumberFormat("fr-FR", {
    style: "percent",
    maximumFractionDigits: 1,
  }).format(value);
}

export function toPayload(form) {
  const optional = (value) => (value === "" ? null : Number(value));
  return {
    amt_income_total: Number(form.amt_income_total),
    amt_credit: Number(form.amt_credit),
    amt_annuity: Number(form.amt_annuity),
    amt_goods_price: Number(form.amt_goods_price),
    age_years: Number(form.age_years),
    employment_years: form.employment_years === "" ? null : Number(form.employment_years),
    ext_source_1: optional(form.ext_source_1),
    ext_source_2: optional(form.ext_source_2),
    ext_source_3: optional(form.ext_source_3),
    name_contract_type: form.name_contract_type,
    name_education_type: form.name_education_type,
    name_family_status: form.name_family_status,
    name_income_type: form.name_income_type,
    cnt_children: Number(form.cnt_children),
    flag_own_realty: form.flag_own_realty,
    flag_own_car: form.flag_own_car,
    region_rating_client: Number(form.region_rating_client),
    code_gender: form.code_gender || null,
  };
}
