"""Colonnes lues dans chaque table et modalités officielles Home Credit.

Seules les colonnes réellement utilisées par le pipeline sont chargées.
"""

APPLICATION_COLUMNS = [
    "SK_ID_CURR",
    "TARGET",
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "REGION_POPULATION_RELATIVE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT",
    "OCCUPATION_TYPE",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
]

BUREAU_COLUMNS = [
    "SK_ID_CURR",
    "SK_ID_BUREAU",
    "CREDIT_ACTIVE",
    "DAYS_CREDIT",
    "CREDIT_DAY_OVERDUE",
    "AMT_CREDIT_SUM",
    "AMT_CREDIT_SUM_DEBT",
    "AMT_CREDIT_SUM_OVERDUE",
]

BUREAU_BALANCE_COLUMNS = [
    "SK_ID_BUREAU",
    "MONTHS_BALANCE",
    "STATUS",
]

PREVIOUS_COLUMNS = [
    "SK_ID_PREV",
    "SK_ID_CURR",
    "AMT_APPLICATION",
    "AMT_CREDIT",
    "NAME_CONTRACT_STATUS",
    "DAYS_DECISION",
]

POS_CASH_COLUMNS = [
    "SK_ID_PREV",
    "SK_ID_CURR",
    "MONTHS_BALANCE",
    "CNT_INSTALMENT",
    "CNT_INSTALMENT_FUTURE",
    "NAME_CONTRACT_STATUS",
    "SK_DPD",
    "SK_DPD_DEF",
]

INSTALLMENTS_COLUMNS = [
    "SK_ID_PREV",
    "SK_ID_CURR",
    "DAYS_INSTALMENT",
    "DAYS_ENTRY_PAYMENT",
    "AMT_INSTALMENT",
    "AMT_PAYMENT",
]

CREDIT_CARD_COLUMNS = [
    "SK_ID_PREV",
    "SK_ID_CURR",
    "MONTHS_BALANCE",
    "AMT_BALANCE",
    "AMT_CREDIT_LIMIT_ACTUAL",
    "AMT_PAYMENT_TOTAL_CURRENT",
    "NAME_CONTRACT_STATUS",
    "SK_DPD",
    "SK_DPD_DEF",
]

TABLE_FILES = {
    "application": "application_train.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "POS_CASH_balance": "POS_CASH_balance.csv",
    "installments_payments": "installments_payments.csv",
    "credit_card_balance": "credit_card_balance.csv",
}

TABLE_COLUMNS = {
    "application": APPLICATION_COLUMNS,
    "bureau": BUREAU_COLUMNS,
    "bureau_balance": BUREAU_BALANCE_COLUMNS,
    "previous_application": PREVIOUS_COLUMNS,
    "POS_CASH_balance": POS_CASH_COLUMNS,
    "installments_payments": INSTALLMENTS_COLUMNS,
    "credit_card_balance": CREDIT_CARD_COLUMNS,
}

CONTRACT_TYPES = ["Cash loans", "Revolving loans"]
GENDERS = ["M", "F", "Non renseigné"]
EDUCATION_TYPES = [
    "Secondary / secondary special",
    "Higher education",
    "Incomplete higher",
    "Lower secondary",
    "Academic degree",
]
FAMILY_STATUSES = [
    "Single / not married",
    "Married",
    "Civil marriage",
    "Widow",
    "Separated",
    "Unknown",
]
INCOME_TYPES = [
    "Working",
    "Commercial associate",
    "Pensioner",
    "State servant",
    "Unemployed",
    "Student",
    "Businessman",
    "Maternity leave",
]
FORM_FAMILY_STATUSES = [value for value in FAMILY_STATUSES if value != "Unknown"]
FORM_INCOME_TYPES = [
    "Working",
    "Commercial associate",
    "Pensioner",
    "State servant",
    "Unemployed",
]
HOUSING_TYPES = [
    "House / apartment",
    "Rented apartment",
    "With parents",
    "Municipal apartment",
    "Office apartment",
    "Co-op apartment",
]
YN = ["Y", "N"]
OCCUPATION_TYPES = [
    "Laborers",
    "Sales staff",
    "Core staff",
    "Managers",
    "Drivers",
    "High skill tech staff",
    "Accountants",
    "Medicine staff",
    "Security staff",
    "Cooking staff",
    "Cleaning staff",
    "Private service staff",
    "Low-skill Laborers",
    "Secretaries",
    "Waiters/barmen staff",
    "Realty agents",
    "HR staff",
    "IT staff",
    "Inconnu",
]

CATEGORY_LEVELS = {
    "NAME_CONTRACT_TYPE": CONTRACT_TYPES,
    "CODE_GENDER": GENDERS,
    "NAME_EDUCATION_TYPE": EDUCATION_TYPES,
    "NAME_FAMILY_STATUS": FAMILY_STATUSES,
    "NAME_INCOME_TYPE": INCOME_TYPES,
    "NAME_HOUSING_TYPE": HOUSING_TYPES,
    "FLAG_OWN_CAR": YN,
    "FLAG_OWN_REALTY": YN,
    "OCCUPATION_TYPE": OCCUPATION_TYPES,
}
