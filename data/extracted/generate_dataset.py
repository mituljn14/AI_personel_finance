#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_dataset.py
===================
Synthetic dataset generator for "AI Personal Finance and Smart Expense Predictor"
(Team 3, WeIntern minor project).

Everything produced here is SYNTHETIC. No real person, account, card, merchant or
transaction is represented. Merchant names are fictional.

Reproducibility: fixed seed (42). Run:   python generate_dataset.py [--out DIR] [--no-ml]
Outputs (in DIR): the 6 CSVs, data_dictionary.csv, dataset_validation_report.txt,
dataset_generation_methodology.md, README.md, generate_dataset.py and a ZIP archive.
Libraries: pandas + numpy for generation/validation/aggregation; scikit-learn only
for the optional forecasting experiment (--no-ml skips it).
"""
import argparse
import calendar
import itertools
import os
import re
import shutil
import sys
import time as _time
import zipfile
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

SEED = 42
N_USERS = 150
START = pd.Timestamp("2022-01-01")
END = pd.Timestamp("2025-12-31")
N_MONTHS = 48
MONTH_STARTS = pd.date_range("2022-01-01", periods=N_MONTHS, freq="MS")
M_YEAR = MONTH_STARTS.year.values
M_MON = MONTH_STARTS.month.values
DAYS_IN = [calendar.monthrange(int(y), int(m))[1] for y, m in zip(M_YEAR, M_MON)]
COUNT_SCALE = 0.55     # realism-preserving calibration: lower values make single entries absurdly large (monthly spend is split across fewer entries)

MONTH_NAMES = list(calendar.month_name)[1:]
EXP_CATS = ["Housing", "Food", "Transportation", "Utilities", "Healthcare", "Education",
            "Shopping", "Entertainment", "Personal Care", "Travel", "Gifts & Donations",
            "Financial Expenses"]


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


# ----------------------------------------------------------------------------
# Calendar knowledge (approximate festival dates, used for seasonality)
# ----------------------------------------------------------------------------
FESTIVALS = {
    "Diwali": ["2022-10-24", "2023-11-12", "2024-11-01", "2025-10-20"],
    "Ganesh Chaturthi": ["2022-08-31", "2023-09-19", "2024-09-07", "2025-08-27"],
    "Durga Puja": ["2022-10-03", "2023-10-22", "2024-10-11", "2025-09-30"],
    "Onam": ["2022-09-08", "2023-08-29", "2024-09-15", "2025-09-05"],
    "Pongal": ["2022-01-14", "2023-01-15", "2024-01-15", "2025-01-14"],
    "Holi": ["2022-03-18", "2023-03-08", "2024-03-25", "2025-03-14"],
    "Dussehra": ["2022-10-05", "2023-10-24", "2024-10-12", "2025-10-02"],
}
REGIONAL = {"Maharashtra": "Ganesh Chaturthi", "Gujarat": "Ganesh Chaturthi", "Goa": "Ganesh Chaturthi",
            "West Bengal": "Durga Puja", "Odisha": "Durga Puja", "Assam": "Durga Puja",
            "Kerala": "Onam", "Tamil Nadu": "Pongal",
            "Delhi": "Holi", "Uttar Pradesh": "Holi", "Rajasthan": "Holi", "Punjab": "Holi",
            "Haryana": "Holi", "Chandigarh": "Holi", "Madhya Pradesh": "Holi", "Bihar": "Holi",
            "Karnataka": "Dussehra", "Telangana": "Dussehra", "Andhra Pradesh": "Dussehra",
            "Chhattisgarh": "Dussehra"}


def _fest_weights():
    W = {}
    for name, ds in FESTIVALS.items():
        w = np.zeros(N_MONTHS)
        for d in pd.to_datetime(ds):
            ws, we = d - pd.Timedelta(days=21), d + pd.Timedelta(days=2)   # 24-day shopping window
            for t, ms in enumerate(MONTH_STARTS):
                me = ms + pd.offsets.MonthEnd(0)
                ov = (min(we, me) - max(ws, ms)).days + 1
                if ov > 0:
                    w[t] += ov / 24.0
        W[name] = w
    return W


FEST_W = _fest_weights()


def festival_index(state):
    """0..1 intensity per month: Diwali (national) + 0.6 x the state's main regional festival."""
    return np.minimum(FEST_W["Diwali"] + 0.6 * FEST_W[REGIONAL.get(state, "Dussehra")], 1.0)


def _norm(tbl):
    a = np.array([tbl[m] for m in M_MON], float)
    mean12 = np.mean([tbl[m] for m in range(1, 13)])
    return a / mean12


TRAVEL_TBL = {1: .9, 2: .8, 3: .9, 4: 1.35, 5: 1.7, 6: 1.5, 7: .8, 8: .75, 9: .75, 10: 1.25, 11: 1.1, 12: 1.7}
ELEC_TBL = {1: .85, 2: .85, 3: .95, 4: 1.2, 5: 1.4, 6: 1.4, 7: 1.1, 8: 1.05, 9: 1.05, 10: 1.0, 11: .9, 12: .85}
ACAD_TBL = {1: 1.4, 2: .75, 3: .75, 4: .75, 5: .75, 6: 2.2, 7: 2.0, 8: 1.3, 9: .75, 10: .75, 11: .75, 12: .75}
BIZ_SEASONS = {
    "festive": {1: .9, 2: .85, 3: .95, 4: .9, 5: .8, 6: .75, 7: .8, 8: .9, 9: 1.1, 10: 1.6, 11: 1.5, 12: 1.0},
    "agri": {1: .7, 2: .8, 3: 1.5, 4: 1.6, 5: 1.0, 6: .7, 7: .6, 8: .7, 9: .9, 10: 1.5, 11: 1.4, 12: .8},
    "tourism": {1: 1.3, 2: 1.0, 3: .9, 4: 1.3, 5: 1.5, 6: 1.2, 7: .6, 8: .6, 9: .7, 10: 1.1, 11: 1.0, 12: 1.8},
}
HOT_STATES = {"Rajasthan", "Delhi", "Uttar Pradesh", "Madhya Pradesh", "Telangana", "Andhra Pradesh",
              "Gujarat", "Punjab", "Haryana", "Chandigarh", "Bihar", "Chhattisgarh", "Odisha", "Maharashtra",
              "Tamil Nadu"}


def seasonal_factor(key, state):
    m = M_MON
    fi = festival_index(state)
    if key is None:
        return np.ones(N_MONTHS)
    if key == "festive_shop":
        return 1 + 0.8 * fi + np.where(np.isin(m, [11, 12, 1, 2]), 0.12, 0.0)   # + wedding season
    if key == "festive_mild":
        return 1 + 0.3 * fi
    if key == "food_festive":
        return 1 + 0.2 * fi
    if key == "restaurant":
        return 1 + 0.2 * fi + np.where(m == 12, 0.1, 0.0)
    if key == "travel":
        return _norm(TRAVEL_TBL)
    if key == "monsoon_taxi":
        return 1 + 0.15 * np.isin(m, [6, 7, 8, 9])
    if key == "monsoon_health":
        return 1 + 0.25 * np.isin(m, [7, 8, 9]) + 0.10 * np.isin(m, [12, 1])
    if key == "academic":
        return _norm(ACAD_TBL)
    raise KeyError(key)


def electricity_factor(state):
    amp = 1.0 if state in HOT_STATES else 0.6
    return 1 + amp * (np.array([ELEC_TBL[m] for m in M_MON]) - 1)


# ----------------------------------------------------------------------------
# Taxonomy
# ----------------------------------------------------------------------------
V = {}   # variable (non-recurring) expense subcategories


def _v(sub, cat, mcat, needs, infl, season, life, unpl, hrs, nouns, w, lam, sig, weekend, desc):
    V[sub] = dict(cat=cat, mcat=mcat, needs=needs, infl=infl, season=season, life=life, unpl=unpl,
                  hrs=hrs, nouns=nouns, w=w, lam=lam, sig=sig, weekend=weekend, desc=desc)


_v("groceries", "Food", "Grocery & Supermarket", True, .055, "food_festive", .35, .05, (8, 21),
   ["Fresh Mart", "Kirana Store", "Supermarket", "Grocers", "Bazaar"], .27, 1.25, .45, .5, "Grocery purchase")
_v("restaurants", "Food", "Restaurant & Cafe", False, .06, "restaurant", .9, .35, (12, 23),
   ["Kitchen", "Cafe", "Dhaba", "Bistro", "Eatery"], .07, .45, .5, 1.5, "Restaurant meal")
_v("food_delivery", "Food", "Food Delivery Service", False, .045, None, .9, .45, (12, 23),
   ["Eats", "Food Express", "Delivery Hub", "Meals Online"], .05, .35, .4, .8, "Food delivery order")
_v("public_transport", "Transportation", "Public Transport Operator", True, .04, None, .2, .05, (6, 20),
   ["Metro Card Recharge", "City Transit", "Bus Pass Office", "Rail Ticket Counter"], .05, .55, .35, 0, "Public transport fare/pass")
_v("fuel", "Transportation", "Fuel Station", True, .03, None, .3, .05, (6, 22),
   ["Fuel Station", "Petrol Pump", "Energy Pump", "Fuel Point"], .10, .55, .3, .3, "Fuel purchase")
_v("taxi", "Transportation", "Ride Service", False, .05, "monsoon_taxi", .8, .45, (6, 23),
   ["Rides", "Cabs", "Go Taxi", "City Cabs"], .04, .30, .5, .5, "Taxi / ride-hailing trip")
_v("vehicle_maintenance", "Transportation", "Auto Service Center", True, .05, None, .5, .35, (9, 18),
   ["Auto Care", "Motor Works", "Garage", "Service Centre"], .025, .06, .5, .2, "Vehicle servicing / repair")
_v("medicine", "Healthcare", "Pharmacy", True, .06, "monsoon_health", .2, .30, (9, 21),
   ["Pharmacy", "Medico", "Chemist", "Health Store"], .04, .25, .5, 0, "Medicines purchase")
_v("consultation", "Healthcare", "Clinic & Hospital", True, .07, "monsoon_health", .2, .70, (9, 19),
   ["Clinic", "Hospital", "Health Centre", "Polyclinic"], .02, .07, .5, 0, "Doctor consultation")
_v("diagnostic_tests", "Healthcare", "Diagnostic Lab", True, .07, "monsoon_health", .2, .50, (7, 13),
   ["Diagnostics", "Pathlab", "Scan Centre"], .015, .04, .5, 0, "Diagnostic tests")
_v("books", "Education", "Bookstore", True, .05, "academic", .3, .10, (10, 20),
   ["Books", "Book Depot", "Stationers"], .008, .04, .4, 0, "Books and stationery")
_v("online_courses", "Education", "Online Learning Platform", False, .03, "academic", .8, .20, (8, 23),
   ["Learn Online", "Skill Academy", "Course Hub"], .012, .035, .4, 0, "Online course fee")
_v("clothing", "Shopping", "Apparel Store", False, .04, "festive_shop", .9, .30, (11, 21),
   ["Apparel", "Fashion Hub", "Garments", "Style Store"], .06, .17, .5, 1.0, "Clothing purchase")
_v("electronics", "Shopping", "Electronics Retailer", False, -.02, "festive_shop", .9, .25, (11, 21),
   ["Electronics", "Digital World", "Gadget Store", "Tech Mart"], .04, .02, .6, .5, "Electronics purchase")
_v("household_goods", "Shopping", "Home & Living Store", True, .04, "festive_mild", .5, .20, (10, 21),
   ["Home Mart", "Living Store", "Household Needs"], .035, .13, .5, .6, "Household goods")
_v("movies", "Entertainment", "Cinema", False, .04, None, .9, .40, (12, 23),
   ["Cinemas", "Multiplex", "Movie Hall"], .012, .13, .35, 1.5, "Movie tickets")
_v("events", "Entertainment", "Events & Ticketing", False, .05, "festive_mild", .9, .45, (15, 23),
   ["Events", "Tickets", "Live Shows"], .01, .04, .5, 1.5, "Event / show tickets")
_v("grooming", "Personal Care", "Salon & Grooming", False, .05, None, .8, .20, (10, 20),
   ["Salon", "Grooming Studio", "Spa & Salon"], .02, .20, .35, .6, "Salon / grooming")
_v("toiletries", "Personal Care", "Personal Care Store", True, .045, None, .3, .10, (9, 21),
   ["Personal Care", "Beauty & Care Store", "Daily Care"], .025, .20, .4, .3, "Toiletries purchase")
_v("train_bus_ticket", "Travel", "Travel Booking", False, .04, "travel", .7, .20, (7, 23),
   ["Travel Desk", "Tickets Online", "Journey Booking"], .018, .06, .4, 0, "Intercity train/bus ticket")
_v("flights", "Travel", "Airline Booking", False, .05, "travel", .9, .25, (7, 23),
   ["Airways", "Air Booking", "Sky Travels"], .03, .025, .5, 0, "Flight booking")
_v("hotel_stay", "Travel", "Hotel & Lodging", False, .05, "travel", .9, .25, (10, 22),
   ["Hotel", "Stay Inn", "Resorts", "Lodge"], .028, .03, .5, 0, "Hotel / lodging")
_v("gifts", "Gifts & Donations", "Gift Shop", False, .04, "festive_shop", .8, .35, (11, 21),
   ["Gifts", "Gift Gallery", "Present Shop"], .02, .07, .5, .5, "Gift purchase")
_v("donations", "Gifts & Donations", "Charitable Trust", False, .03, "festive_mild", .5, .20, (9, 20),
   ["Charitable Trust", "Relief Foundation", "Community Fund"], .01, .035, .5, 0, "Donation")
_v("home_repairs", "Housing", "Home Services", True, .05, None, .5, .60, (9, 18),
   ["Home Services", "Repair Works", "Handyman Co"], .02, .035, .55, .3, "Home repair / service")

# recurring-type expense subcategories: sub -> (category, merchant_category, needs)
REC_SUB = {
    "rent": ("Housing", "Landlord / Property Owner", True),
    "maintenance": ("Housing", "Housing Society", True),
    "electricity": ("Utilities", "Electricity Distribution", True),
    "water": ("Utilities", "Water Board", True),
    "gas": ("Utilities", "Gas Supplier", True),
    "internet": ("Utilities", "Broadband Provider", True),
    "mobile_recharge": ("Utilities", "Telecom Provider", True),
    "subscriptions": ("Entertainment", "Streaming & Subscriptions", False),
    "insurance": ("Healthcare", "Insurance Provider", True),
    "vehicle_insurance": ("Transportation", "Insurance Provider", True),
    "loan_payment": ("Financial Expenses", "Lender", True),
    "tuition": ("Education", "School / College", True),
    "bank_fees": ("Financial Expenses", "Bank", True),
    "interest_charges": ("Financial Expenses", "Bank", True),
}
SAV_SUBS = {"sip_mutual_fund": "Mutual fund SIP contribution", "recurring_deposit": "Recurring deposit instalment",
            "ppf_contribution": "PPF annual contribution", "fixed_deposit": "Fixed deposit placement",
            "savings_withdrawal": "Redemption / maturity credited to account"}
TRF_SUBS = {"own_account_transfer": "Transfer between own accounts", "atm_cash_withdrawal": "ATM cash withdrawal"}
INC_SUBS = ["salary", "freelance_work", "business_income", "incentives", "other_income"]

TAXONOMY = []   # (category, subcategory, transaction types, essential?)
for s, a in V.items():
    TAXONOMY.append((a["cat"], s, "expense/refund", a["needs"]))
for s, (c, mc, n) in REC_SUB.items():
    TAXONOMY.append((c, s, "expense/refund", n))
for s in INC_SUBS:
    TAXONOMY.append(("Income", s, "income", None))
for s in SAV_SUBS:
    TAXONOMY.append(("Savings and Investments", s, "transfer", None))
for s in TRF_SUBS:
    TAXONOMY.append(("Transfers", s, "transfer", None))
VALID_PAIRS = {(c, s) for c, s, _, _ in TAXONOMY}
SUB_CAT = {s: c for c, s, _, _ in TAXONOMY}
SUB_NEEDS = {s: n for _, s, _, n in TAXONOMY}

PREFIXES = ["Sunrise", "Urban", "Prime", "Evergreen", "Royal", "Metro", "Lotus", "Heritage", "Zenith", "Orchid",
            "Pioneer", "Skyline", "Maple", "Harbor", "Silverline", "Crescent", "Emerald", "Vista", "Nimbus", "Banyan"]
_mrng = np.random.default_rng(SEED + 1)
MERCH_POOLS = {}
for _s, _a in V.items():
    cand = [f"{p} {n}" for p in PREFIXES for n in _a["nouns"]]
    MERCH_POOLS[_s] = [cand[i] for i in _mrng.choice(len(cand), size=10, replace=False)]
REC_MERCH = {
    "internet": ["Skyline Broadband", "BlueWave Fibre", "NetSpark Broadband", "Orbit Fibre"],
    "mobile_recharge": ["Zenith Mobile", "Pulse Telecom", "Aero Mobile"],
    "subscriptions": ["StreamBox", "CineFlix Plus", "MelodyStream", "WatchWave"],
    "insurance": ["Shield Life Insurance Co", "SafeGuard Health Insurance", "Assure General Insurance"],
    "vehicle_insurance": ["Assure General Insurance", "RoadSafe Motor Insurance"],
    "loan_payment": ["Heritage Housing Finance", "Unity Bank Loans", "Apex Vehicle Finance", "Crest Personal Finance"],
    "tuition": ["Little Scholars School", "Bright Future Academy", "Learn Hub Classes", "Sunrise Public School"],
    "maintenance": ["Green Meadows Housing Society", "Skyview Apartments Association", "Lakeside Residents Welfare"],
    "sip_mutual_fund": ["Nova Mutual Fund AMC", "Pinnacle Asset Management", "Horizon Funds AMC"],
    "recurring_deposit": ["Unity Bank Recurring Deposit", "Civic Bank Recurring Deposit"],
    "ppf_contribution": ["Public Provident Fund Account"],
    "fixed_deposit": ["Unity Bank Fixed Deposit", "Civic Bank Fixed Deposit"],
    "savings_withdrawal": ["Unity Bank Fixed Deposit", "Nova Mutual Fund AMC"],
    "bank_fees": ["Unity Bank Charges", "Civic Bank Charges"],
    "interest_charges": ["Credit Card Interest and Charges"],
}
SAV_MCAT = {"sip_mutual_fund": "Mutual Fund", "recurring_deposit": "Bank Deposit", "ppf_contribution": "Government Savings Scheme",
            "fixed_deposit": "Bank Deposit", "savings_withdrawal": "Investment Redemption"}

# ----------------------------------------------------------------------------
# Payment methods, accounts, channels
# ----------------------------------------------------------------------------
M_ORDER = ["UPI", "Debit Card", "Credit Card", "Cash", "Net Banking", "Bank Transfer", "Digital Wallet", "Auto Debit"]
BASE_W = np.array([.46, .13, .13, .14, .03, 0., .06, 0.])
SUB_MULT = {
    "groceries": {"Cash": 1.3, "Digital Wallet": .8}, "restaurants": {"Credit Card": 1.2, "Cash": .5},
    "food_delivery": {"Cash": 0.0, "Debit Card": .6, "Credit Card": 1.2, "Digital Wallet": 1.3},
    "public_transport": {"Cash": 1.6, "Digital Wallet": 1.5, "Credit Card": 0.0},
    "fuel": {"Cash": 1.1, "Credit Card": 1.4, "Debit Card": 1.2}, "taxi": {"Cash": 1.2, "UPI": 1.2, "Credit Card": .4},
    "vehicle_maintenance": {"Cash": 1.5}, "medicine": {"Cash": 1.3}, "consultation": {"Cash": 1.3},
    "flights": {"Cash": 0.0, "Digital Wallet": 0.0, "Credit Card": 2.0, "Net Banking": 3.0},
    "hotel_stay": {"Cash": .3, "Credit Card": 1.8, "Net Banking": 2.0},
    "train_bus_ticket": {"Cash": .1, "Credit Card": 1.2, "Net Banking": 2.0},
    "electronics": {"Cash": .2, "Credit Card": 2.0, "Net Banking": 2.0},
    "online_courses": {"Cash": 0.0, "Credit Card": 1.5, "Net Banking": 2.0},
    "clothing": {"Credit Card": 1.4}, "gifts": {"Cash": 1.0}, "donations": {"Cash": 1.5, "UPI": 1.2},
    "events": {"Cash": .2, "Credit Card": 1.2}, "movies": {"Cash": .2},
    "home_repairs": {"Cash": 1.8}, "bank_fees": {"Cash": 0.0},
}
METHOD_ACCOUNT = {"UPI": "savings_account", "Debit Card": "savings_account", "Credit Card": "credit_card",
                  "Cash": "cash_in_hand", "Net Banking": "savings_account", "Bank Transfer": "savings_account",
                  "Digital Wallet": "digital_wallet", "Auto Debit": "savings_account"}
METHOD_SOURCE = {"UPI": "upi_app_export", "Debit Card": "card_statement", "Credit Card": "card_statement",
                 "Cash": "manual_entry", "Net Banking": "bank_statement_import", "Bank Transfer": "bank_statement_import",
                 "Digital Wallet": "wallet_export", "Auto Debit": "bank_statement_import"}
ONLINE_SUBS = {"food_delivery", "online_courses", "flights", "hotel_stay", "train_bus_ticket", "taxi", "internet",
               "mobile_recharge", "subscriptions", "electricity", "water", "gas"}
ONLINE_P = {"electronics": .6, "clothing": .4, "groceries": .15, "gifts": .4, "books": .4, "events": .8,
            "movies": .5, "medicine": .1}

# ----------------------------------------------------------------------------
# Profiles, geography, occupations
# ----------------------------------------------------------------------------
PROFILE = {
    "disciplined_saver": dict(ratio=(0.50, 0.68), save=(0.12, 0.25), p_sip=.85, bias=(1.00, 1.12), sigma=(0.05, 0.10),
                              stability=(0.75, 0.95), unpl=.8),
    "balanced": dict(ratio=(0.68, 0.84), save=(0.05, 0.12), p_sip=.5, bias=(0.97, 1.10), sigma=(0.08, 0.15),
                     stability=(0.5, 0.8), unpl=1.0),
    "overspender": dict(ratio=(0.84, 0.95), save=(0.01, 0.05), p_sip=.15, bias=(0.75, 0.90), sigma=(0.12, 0.20),
                        stability=(0.3, 0.6), unpl=1.4),
    "inconsistent": dict(ratio=(0.62, 0.90), save=(0.02, 0.10), p_sip=.3, bias=(0.85, 1.05), sigma=(0.20, 0.32),
                         stability=(0.15, 0.4), unpl=1.2),
}
CITIES = [("Mumbai", "Maharashtra", 1, 10), ("Pune", "Maharashtra", 1, 10), ("Nagpur", "Maharashtra", 2, 4),
          ("Nashik", "Maharashtra", 2, 3), ("Kolhapur", "Maharashtra", 3, 2), ("Delhi", "Delhi", 1, 9),
          ("Bengaluru", "Karnataka", 1, 9), ("Mysuru", "Karnataka", 3, 2), ("Hyderabad", "Telangana", 1, 7),
          ("Chennai", "Tamil Nadu", 1, 6), ("Coimbatore", "Tamil Nadu", 2, 3), ("Kolkata", "West Bengal", 1, 6),
          ("Ahmedabad", "Gujarat", 1, 5), ("Surat", "Gujarat", 2, 3), ("Vadodara", "Gujarat", 2, 2),
          ("Jaipur", "Rajasthan", 2, 4), ("Lucknow", "Uttar Pradesh", 2, 4), ("Indore", "Madhya Pradesh", 2, 3),
          ("Bhopal", "Madhya Pradesh", 2, 2), ("Kochi", "Kerala", 2, 3), ("Thiruvananthapuram", "Kerala", 2, 2),
          ("Chandigarh", "Chandigarh", 2, 2), ("Patna", "Bihar", 2, 2), ("Bhubaneswar", "Odisha", 2, 2),
          ("Guwahati", "Assam", 2, 2), ("Raipur", "Chhattisgarh", 3, 1), ("Ludhiana", "Punjab", 2, 2),
          ("Gurugram", "Haryana", 1, 3), ("Visakhapatnam", "Andhra Pradesh", 2, 2), ("Panaji", "Goa", 3, 1)]
OCC = {
    "Salaried": [("Software Engineer", 70000, .35), ("Teacher", 32000, .25), ("Bank Officer", 55000, .25),
                 ("Government Clerk", 30000, .2), ("Nurse", 28000, .25), ("Accountant", 38000, .25),
                 ("Marketing Executive", 40000, .3), ("Mechanical Engineer", 45000, .3), ("Sales Executive", 30000, .3),
                 ("HR Executive", 38000, .25), ("Data Analyst", 62000, .3), ("Pharmacist", 35000, .25),
                 ("Government Officer", 65000, .25), ("Operations Manager", 85000, .3)],
    "Freelancer": [("Freelance Designer", 45000, .4), ("Freelance Writer", 30000, .4), ("Consultant", 80000, .4),
                   ("Photographer", 35000, .4), ("Freelance Web Developer", 60000, .4), ("Private Tutor", 25000, .3)],
    "Business Owner": [("Retail Shop Owner", 50000, .4), ("Restaurant Owner", 70000, .4), ("Small Trader", 45000, .4),
                       ("Contractor", 65000, .4), ("Transport Operator", 55000, .35), ("Agri-business Owner", 50000, .4)],
    "Gig/Part-time": [("Delivery Partner", 18000, .2), ("Cab Driver", 24000, .2), ("Part-time Worker", 14000, .2)],
    "Student": [("Student", 12000, .35)],
    "Retired": [("Retired (Pension)", 30000, .4)],
}
AGE_RANGE = {"Salaried": (23, 32, 58), "Freelancer": (24, 33, 52), "Business Owner": (27, 40, 64),
             "Gig/Part-time": (20, 28, 46), "Student": (19, 21, 25), "Retired": (60, 65, 78)}


def build_users(grng):
    types = ["Salaried"] * 80 + ["Freelancer"] * 20 + ["Business Owner"] * 18 + ["Gig/Part-time"] * 10 + \
            ["Student"] * 10 + ["Retired"] * 12
    grng.shuffle(types)
    profs = ["disciplined_saver"] * 38 + ["balanced"] * 52 + ["overspender"] * 30 + ["inconsistent"] * 30
    grng.shuffle(profs)
    cw = np.array([c[3] for c in CITIES], float)
    cw /= cw.sum()
    users = []
    for i in range(N_USERS):
        et = types[i]
        occ, med, sg = OCC[et][int(grng.integers(len(OCC[et])))]
        lo, mode, hi = AGE_RANGE[et]
        age = int(round(grng.triangular(lo, mode, hi)))
        city, state, tier, _ = CITIES[int(grng.choice(len(CITIES), p=cw))]
        inc = float(grng.lognormal(np.log(med), sg))
        if et == "Salaried":
            inc *= max(0.8, 1 + 0.012 * (age - 25))
            inc = float(np.clip(inc, 15000, 220000))
        elif et == "Student":
            inc = float(np.clip(inc, 6000, 30000))
        elif et == "Retired":
            inc = float(np.clip(inc, 12000, 90000))
        else:
            inc = float(np.clip(inc, 8000, 200000))
        inc = float(round(inc / 500) * 500)
        if et == "Student":
            hh = int(grng.choice([1, 2], p=[.85, .15]))
        elif et == "Retired":
            hh = int(grng.choice([1, 2, 3], p=[.2, .6, .2]))
        elif age < 28:
            hh = int(grng.choice([1, 2, 3], p=[.5, .3, .2]))
        elif age < 40:
            hh = int(grng.choice([1, 2, 3, 4, 5], p=[.1, .25, .3, .25, .1]))
        else:
            hh = int(grng.choice([2, 3, 4, 5, 6], p=[.15, .25, .3, .2, .1]))
        prof = profs[i]
        P = PROFILE[prof]
        has_cc = bool(grng.random() < (0.1 if et == "Student" else 0.3 if et == "Retired" else
                                       float(np.clip((inc - 20000) / 80000, 0, .85))))
        cash_pref = float(np.clip(.10 + (.1 if age > 50 else 0) + (.1 if tier == 3 else 0) +
                                  (.1 if et in ("Business Owner", "Gig/Part-time") else 0) +
                                  (.08 if inc < 25000 else 0), .05, .45))
        veh_p = 0.05 if age < 22 else float(np.clip(.15 + inc / 150000 * .6, .05, .85))
        if et == "Student":
            veh_p = .12
        vehicle = bool(grng.random() < veh_p)
        rent_p = {"Student": .85, "Retired": .1, "Business Owner": .35, "Freelancer": .5, "Gig/Part-time": .7}.get(
            et, .75 if age < 35 else .45 if age < 50 else .25)
        pref = ["UPI", "Debit Card", "Credit Card", "Cash", "Net Banking"][
            int(grng.choice(5, p=[.58, .14, .12, .10, .06]))]
        if pref == "Credit Card" and not has_cc:
            pref = "UPI"
        shift_t, shift_mult = None, 1.0
        if grng.random() < .15:
            shift_t = int(grng.integers(12, 36))
            shift_mult = float(grng.choice([grng.uniform(.75, .85), grng.uniform(1.2, 1.4)]))
        users.append(dict(
            idx=i + 1, user_id=f"U{i + 1:04d}", employment_type=et, occupation=occ, age=age, hh=hh, city=city,
            state=state, tier=tier, income_base=inc, profile=prof,
            income_frequency="monthly" if et in ("Salaried", "Retired", "Student") else
            ("irregular" if et in ("Freelancer", "Gig/Part-time") else str(grng.choice(["irregular", "seasonal"]))),
            ratio=float(grng.uniform(*P["ratio"])), save_frac=float(grng.uniform(*P["save"])),
            sigma=float(grng.uniform(*P["sigma"])), stability=float(grng.uniform(*P["stability"])),
            bias=float(grng.uniform(*P["bias"])), unpl=P["unpl"], has_cc=has_cc, pref_method=pref,
            cash_pref=cash_pref, vehicle=vehicle, vehicle_car=bool(vehicle and inc > 60000 and grng.random() < .35),
            renter=bool(grng.random() < rent_p),
            kids=bool(hh >= 3 and 30 <= age <= 55 and grng.random() < .6),
            taste={s: float(grng.lognormal(0, .3)) for s in V}, foodie=float(grng.lognormal(0, .35)),
            delivery=float(0.2 if et == "Retired" else (1.6 if age < 30 else 1.0) * (0.5 if tier == 3 else 1.0) *
                           grng.lognormal(0, .3)),
            freq_mult=float(grng.lognormal(0, .25)), amt_disp=float(grng.uniform(.8, 1.3)),
            shift_t=shift_t, shift_mult=shift_mult, atm_logged=bool(grng.random() < .5),
            other_income=bool(grng.random() < .25), g_user=float(np.clip(grng.normal(.05, .04), -.03, .15)),
            biz_season=str(grng.choice(list(BIZ_SEASONS))), opening_mult=float(grng.uniform(.8, 4.0)),
            profile_created=pd.Timestamp("2021-10-01") + pd.Timedelta(days=int(grng.integers(0, 93))),
        ))
    return users


# ----------------------------------------------------------------------------
# Per-user generation
# ----------------------------------------------------------------------------
PERIOD = {"monthly": 1, "bimonthly": 2, "quarterly": 3, "half_yearly": 6, "annual": 12}
WEEKDAYS = [np.array([pd.Timestamp(int(M_YEAR[t]), int(M_MON[t]), d).weekday() for d in range(1, DAYS_IN[t] + 1)])
            for t in range(N_MONTHS)]


def rand_time(rng, hrs):
    h = min(int(rng.integers(hrs[0], hrs[1] + 1)), 23)
    return f"{h:02d}:{int(rng.integers(0, 60)):02d}:{int(rng.integers(0, 60)):02d}"


def pick_day(rng, t, bias):
    w = 1.0 + bias * (WEEKDAYS[t] >= 5)
    return int(rng.choice(DAYS_IN[t], p=w / w.sum())) + 1


def sample_count(rng, lam, stability):
    if lam <= 0:
        return 0
    if rng.random() < stability:
        return int(lam) + (1 if rng.random() < lam - int(lam) else 0)
    return int(rng.poisson(lam))


def round_amt(x, base=1):
    return float(max(base, int(round(x / base)) * base)) if base > 1 else float(max(1, int(round(x))))


def gen_user(u, rng, keygen):
    uid, et, state = u["user_id"], u["employment_type"], u["state"]
    rows, items, budgets = [], [], []
    base_inc = u["income_base"]
    tier_f = {1: 1.15, 2: 1.0, 3: 0.85}[u["tier"]]
    fav = {}
    seq = itertools.count(1)
    cash_tot = np.zeros(N_MONTHS)
    P = PROFILE[u["profile"]]

    def merchant(sub, pool=None):
        pool = pool or MERCH_POOLS.get(sub) or REC_MERCH[sub]
        if sub not in fav:
            fav[sub] = [pool[i] for i in rng.choice(len(pool), size=min(3, len(pool)), replace=False)]
        f = fav[sub]
        return f[int(rng.choice(len(f), p=[.6, .3, .1][:len(f)] if len(f) == 3 else [1 / len(f)] * len(f)))]

    def pick_method(sub, amount, allowed=None):
        w = BASE_W.copy()
        w[M_ORDER.index(u["pref_method"])] *= 3
        w[3] *= u["cash_pref"] / 0.14
        if not u["has_cc"]:
            w[2] = 0
        for k, v in SUB_MULT.get(sub, {}).items():
            w[M_ORDER.index(k)] *= v
        if amount > 100000:
            w[0] = 0
        if amount >= 50000:
            w[[3, 6]] = 0
            w[[4, 5]] += .08
        elif amount >= 20000:
            w[3] *= .1
            w[6] *= .2
        if amount < 150:
            w[[2, 4]] *= .1
        if allowed is not None:
            mask = np.array([m in allowed for m in M_ORDER], float)
            w = np.array([allowed[m] if m in allowed else 0 for m in M_ORDER], float) if isinstance(allowed, dict) else w * mask
            if "Cash" in M_ORDER and isinstance(allowed, dict) and "Cash" in allowed:
                w[3] = allowed["Cash"] * u["cash_pref"] / 0.14
            if not u["has_cc"]:
                w[2] = 0
        if w.sum() <= 0:
            w = np.zeros(8)
            w[0] = 1
        return M_ORDER[int(rng.choice(8, p=w / w.sum()))]

    def emit(date, typ, amount, sub, desc, method, merch, mcat, *, rid=None, freq=None, planned="not_applicable",
             needs=None, status="completed", time=None, related=None, city=None, st=None, channel=None,
             account=None, source=None, category=None):
        cat = category or SUB_CAT[sub]
        if needs is None:
            needs = ("needs" if SUB_NEEDS[sub] else "wants") if typ in ("expense", "refund") else \
                ("savings" if cat == "Savings and Investments" else "not_applicable")
        if channel is None:
            if method == "Cash":
                channel = "cash_counter"
            elif method == "Auto Debit":
                channel = "auto_debit"
            elif method in ("Net Banking", "Bank Transfer"):
                channel = "bank_portal"
            else:
                channel = "online" if (sub in ONLINE_SUBS or rng.random() < ONLINE_P.get(sub, 0)) else "in_store"
        if account is None:
            account = METHOD_ACCOUNT[method]
            if et == "Business Owner" and typ == "income":
                account = "current_account"
        if source is None:
            source = METHOD_SOURCE[method]
        if time is None:
            time = ""
        if method == "Cash" and merch and rng.random() < .35 and typ == "expense" and rid is None:
            merch = ""        # unknown merchant for informal cash payments
        if source == "manual_entry" and time and rng.random() < .5:
            time = ""         # manual entries often carry no time-of-day
        k = next(keygen)
        rows.append(dict(key=k, user_id=uid, date=date, time=time, transaction_type=typ, amount=round(float(amount), 2),
                         category=cat, subcategory=sub, description=desc, payment_method=method, merchant_name=merch,
                         merchant_category=mcat, account_type=account, is_recurring=int(rid is not None),
                         recurring_frequency=freq or "", essential=int(needs == "needs" and typ in ("expense", "refund")),
                         planned=planned, status=status, channel=channel, city=city or u["city"], state=st or state,
                         needs=needs, source=source, recurring_id=rid or "", related_key=related))
        if typ == "expense" and method == "Cash" and status == "completed":
            cash_tot[(date.year - 2022) * 12 + date.month - 1] += amount
        return k

    # ------------------------------------------------------------ recurring items
    def new_item(typ, sub, desc, freq, day, t0, base, *, t_last=47, gmean=0., gsd=0., cm=None, p_change=.85, extras=(),
                 rnd=1, kind=None, jit=0, methods=None, merch=None, merch2=None, switch=None, mcat=None, essential=None):
        per = PERIOD[freq]
        ch = list(extras)
        if cm:
            for y in (2023, 2024, 2025):
                d = pd.Timestamp(y, cm, 1)
                if d > pd.Timestamp(int(M_YEAR[t0]), int(M_MON[t0]), 1) and rng.random() < p_change:
                    ch.append((d, 1 + max(0., float(rng.normal(gmean, gsd)))))
        ch.sort(key=lambda z: z[0])

        def amt_at(d):
            f = 1.0
            for cd, fa in ch:
                if cd <= d:
                    f *= fa
            return base * f
        ts = list(range(t0, t_last + 1, per))
        dates = [pd.Timestamp(int(M_YEAR[t]), int(M_MON[t]), day) for t in ts]
        amts = [round_amt(amt_at(d), rnd) if rnd >= 1 else round(amt_at(d), 2) for d in dates]
        it = dict(typ=typ, sub=sub, desc=desc, freq=freq, per=per, day=day, ts=ts, dates=dates, amts=amts, kind=kind,
                  jit=jit, methods=methods, merch=merch, merch2=merch2, switch=switch, mcat=mcat, amt_at=amt_at,
                  base=base, ongoing=(t_last == 47), t0=t0, t_last=t_last, essential=essential)
        items.append(it)
        return it

    # ---- income
    inc_exp = np.zeros(N_MONTHS)
    ms = list(MONTH_STARTS)
    salary_item = None
    if et == "Salaried":
        extras = []
        sw = None
        if rng.random() < .18:
            jt = int(rng.integers(6, 42))
            extras.append((ms[jt], 1 + float(rng.uniform(.12, .35))))
            sw = (ms[jt], "{} {} Pvt Ltd".format(rng.choice(PREFIXES), rng.choice(["Systems", "Solutions", "Industries", "Services"])))
        employer = "{} {} Pvt Ltd".format(rng.choice(PREFIXES), rng.choice(["Systems", "Solutions", "Industries", "Services"]))
        if u["occupation"] in ("Government Clerk", "Government Officer"):
            employer = "State Government Department (synthetic)"
        salary_item = new_item("income", "salary", "Monthly salary credit", "monthly", int(rng.choice([1, 1, 1, 5, 7, 10, 25, 28])), 0,
                               base_inc, gmean=.07, gsd=.03, cm=int(rng.choice([4, 4, 4, 1, 7])), p_change=.9, extras=extras,
                               jit=1, methods={"Bank Transfer": 1.0}, merch=employer, merch2=sw[1] if sw else None,
                               switch=sw[0] if sw else None, mcat="Employer")
        inc_exp = np.array([salary_item["amt_at"](d) for d in ms])
    elif et == "Retired":
        it = new_item("income", "other_income", "Monthly pension credit", "monthly", int(rng.integers(1, 6)), 0, base_inc,
                      gmean=.04, gsd=.01, cm=7, jit=0, methods={"Bank Transfer": 1.0}, merch="Pension Disbursement Office (synthetic)",
                      mcat="Pension Fund")
        inc_exp = np.array([it["amt_at"](d) for d in ms])
        if rng.random() < .25:
            new_item("income", "other_income", "Rental income", "monthly", 5, 0, float(rng.uniform(5000, 18000)), gmean=.05, gsd=.02,
                     cm=4, jit=1, methods={"Bank Transfer": .6, "UPI": .4}, merch="Tenant-" + uid[-4:], mcat="Tenant")
    elif et == "Student":
        it = new_item("income", "other_income", "Monthly allowance from family", "monthly", int(rng.integers(1, 8)), 0, base_inc,
                      gmean=.06, gsd=.03, cm=6, p_change=.6, jit=1, methods={"UPI": .6, "Bank Transfer": .4},
                      merch="Family Transfer", mcat="Family")
        inc_exp = np.array([it["amt_at"](d) for d in ms])
    else:
        seas = np.ones(N_MONTHS)
        if u["income_frequency"] == "seasonal":
            seas = _norm(BIZ_SEASONS[u["biz_season"]])
        inc_exp = base_inc * (1 + u["g_user"]) ** (np.arange(N_MONTHS) / 12) * seas
    retainer = None
    if et == "Freelancer" and rng.random() < .45:
        retainer = new_item("income", "freelance_work", "Retainer from long-term client", "monthly", int(rng.integers(5, 26)), 0,
                            .3 * base_inc, gmean=.05, gsd=.03, cm=int(rng.integers(1, 13)), jit=1,
                            methods={"Bank Transfer": .7, "UPI": .3}, merch="Client-R" + uid[-3:], mcat="Client")
    if et == "Business Owner" and rng.random() < .15:
        new_item("income", "other_income", "Rental income", "monthly", 5, 0, float(rng.uniform(5000, 18000)), gmean=.05, gsd=.02, cm=4,
                 jit=1, methods={"Bank Transfer": .6, "UPI": .4}, merch="Tenant-" + uid[-4:], mcat="Tenant")

    # ---- housing / utilities / subscriptions / insurance / loans / education
    if u["renter"]:
        if et == "Student":
            rent = min(float(rng.uniform(3500, 9000)), .45 * base_inc)
        else:
            rent = float(np.clip(base_inc * rng.uniform(.17, .30) * tier_f, 3500, .32 * base_inc))
        new_item("expense", "rent", "Monthly house rent", "monthly", int(rng.integers(1, 11)), 0, rent, gmean=.06, gsd=.025,
                 cm=int(rng.integers(1, 13)), p_change=.8, rnd=100 if rent > 5000 else 50, jit=1,
                 methods={"Bank Transfer": .45, "UPI": .25, "Net Banking": .2, "Cash": u["cash_pref"] if u["tier"] == 3 else .03},
                 merch="Landlord-" + uid[-4:], mcat="Landlord / Property Owner")
    if rng.random() < (.5 if u["renter"] or et != "Student" else .1):
        new_item("expense", "maintenance", "Society maintenance charges", "monthly", int(rng.integers(1, 11)), 0,
                 float(rng.uniform(1000, 5000) * tier_f), gmean=.05, gsd=.02, cm=4, rnd=50, jit=1,
                 methods={"UPI": .45, "Net Banking": .35, "Bank Transfer": .2}, merch=None)
    elec_f = electricity_factor(state)
    e_base = (320 + 110 * u["hh"]) * tier_f * (1 + base_inc / 200000) * float(rng.uniform(.85, 1.2))
    e_freq = "monthly" if rng.random() < .6 else "bimonthly"
    new_item("expense", "electricity", "Electricity bill", e_freq, int(rng.integers(5, 26)), int(rng.integers(0, 2)),
             e_base * PERIOD[e_freq], gmean=.05, gsd=.02, cm=int(rng.integers(1, 13)), rnd=0, kind="electricity", jit=1,
             methods={"UPI": .5, "Net Banking": .15, "Auto Debit": .2, "Credit Card": .1, "Digital Wallet": .05},
             merch="{} Power Distribution Co".format(state))
    if rng.random() < .4:
        new_item("expense", "water", "Water bill", "quarterly", int(rng.integers(5, 26)), int(rng.integers(0, 3)),
                 float(rng.uniform(250, 800) * (1 + .1 * u["hh"])), gmean=.04, gsd=.02, cm=4, rnd=0, kind="water", jit=1,
                 methods={"UPI": .6, "Net Banking": .4}, merch="{} Municipal Water Board".format(u["city"]))
    r = rng.random()
    if r < .25:
        new_item("expense", "gas", "Piped gas bill", "monthly", int(rng.integers(5, 26)), 0, float(rng.uniform(350, 900)),
                 gmean=.04, gsd=.02, cm=4, rnd=0, kind="gas", jit=1, methods={"UPI": .6, "Net Banking": .4},
                 merch="{} Gas Distribution".format(u["city"]))
    elif r < .55:
        new_item("expense", "gas", "LPG cylinder refill", "bimonthly", int(rng.integers(5, 26)), int(rng.integers(0, 2)),
                 float(rng.uniform(850, 1100)), gmean=.04, gsd=.02, cm=4, rnd=0, kind="gas", jit=1,
                 methods={"UPI": .6, "Cash": .25 + u["cash_pref"], "Digital Wallet": .15}, merch="{} Gas Distribution".format(u["city"]))
    if rng.random() < (.35 if u["tier"] == 3 else .5):
        new_item("expense", "internet", "Broadband bill", "monthly", int(rng.integers(3, 20)), 0,
                 float(rng.choice([499, 599, 799, 999, 1199])), gmean=.04, gsd=.02, cm=int(rng.integers(1, 13)), p_change=.4,
                 jit=1, methods={"UPI": .55, "Auto Debit": .2, "Credit Card": .15, "Digital Wallet": .1}, merch=None)
    r = rng.random()
    if r < .3:
        new_item("expense", "mobile_recharge", "Postpaid mobile bill", "monthly", int(rng.integers(3, 20)), 0,
                 float(rng.choice([299, 399, 499, 699, 999])), gmean=.10, gsd=.04, cm=int(rng.integers(1, 13)), p_change=.5, jit=1,
                 methods={"UPI": .5, "Auto Debit": .3, "Credit Card": .2}, merch=None)
    elif r < .7:
        new_item("expense", "mobile_recharge", "Prepaid mobile recharge (84-day pack)", "quarterly", int(rng.integers(3, 25)),
                 int(rng.integers(0, 3)), float(rng.choice([599, 719, 839])), gmean=.10, gsd=.04, cm=int(rng.integers(1, 13)),
                 p_change=.5, jit=1, methods={"UPI": .7, "Digital Wallet": .2, "Cash": .1}, merch=None)
    else:
        new_item("expense", "mobile_recharge", "Prepaid mobile recharge (56-day pack)", "bimonthly", int(rng.integers(3, 25)),
                 int(rng.integers(0, 2)), float(rng.choice([399, 479, 599])), gmean=.10, gsd=.04, cm=int(rng.integers(1, 13)),
                 p_change=.5, jit=1, methods={"UPI": .7, "Digital Wallet": .2, "Cash": .1}, merch=None)
    if rng.random() < .45:
        late = rng.random() < .25
        new_item("expense", "subscriptions", "Streaming subscription", "monthly", int(rng.integers(1, 26)),
                 int(rng.integers(1, 40)) if late else 0, float(rng.choice([149, 199, 299, 499, 649])), gmean=.07, gsd=.03,
                 cm=int(rng.integers(1, 13)), p_change=.5, t_last=int(rng.integers(30, 47)) if rng.random() < .12 else 47,
                 jit=0, methods={"Credit Card": .45, "UPI": .35, "Auto Debit": .2}, merch=None)
    if et not in ("Student",) and rng.random() < .55:
        new_item("expense", "insurance", "Health insurance annual premium", "annual", int(rng.integers(1, 26)), int(rng.integers(0, 12)),
                 float((4000 + 4500 * u["hh"]) * rng.uniform(.8, 1.4)), gmean=.08, gsd=.03, cm=None, rnd=100, jit=0,
                 methods={"Net Banking": .5, "Auto Debit": .2, "UPI": .2, "Credit Card": .1}, merch=None)
    if et in ("Salaried", "Business Owner", "Freelancer") and 25 <= u["age"] <= 55 and rng.random() < .3:
        new_item("expense", "insurance", "Term life insurance annual premium", "annual", int(rng.integers(1, 26)), int(rng.integers(0, 12)),
                 float(base_inc * 12 * rng.uniform(.01, .025)), gmean=.0, gsd=.0, cm=None, rnd=100, jit=0,
                 methods={"Net Banking": .5, "Auto Debit": .3, "UPI": .2}, merch=None)
    if u["vehicle"] and rng.random() < .7:
        new_item("expense", "vehicle_insurance", "Vehicle insurance annual premium", "annual", int(rng.integers(1, 26)),
                 int(rng.integers(0, 12)), float(rng.uniform(6000, 14000) if u["vehicle_car"] else rng.uniform(1200, 2800)),
                 gmean=.06, gsd=.02, cm=None, rnd=50, jit=0, methods={"Net Banking": .5, "UPI": .3, "Credit Card": .2}, merch=None)
    emi_total = 0.0
    if et in ("Salaried", "Business Owner", "Freelancer") and u["age"] >= 28 and rng.random() < .17:
        emi = base_inc * rng.uniform(.12, .25)
        emi_total += emi
        new_item("expense", "loan_payment", "Home loan EMI", "monthly", int(rng.integers(1, 11)), int(rng.integers(0, 6)), emi,
                 rnd=100, jit=0, methods={"Auto Debit": .75, "Net Banking": .25}, merch=None)
    if u["vehicle"] and rng.random() < .15 and emi_total < .2 * base_inc:
        t0 = int(rng.integers(0, 20))
        emi = base_inc * rng.uniform(.06, .11)
        emi_total += emi
        new_item("expense", "loan_payment", "Vehicle loan EMI", "monthly", int(rng.integers(1, 11)), t0, emi, rnd=100,
                 t_last=min(47, t0 + int(rng.choice([36, 48, 60])) - 1), jit=0,
                 methods={"Auto Debit": .75, "Net Banking": .25}, merch=None)
    if et != "Retired" and rng.random() < .1 and emi_total < .25 * base_inc:
        t0 = int(rng.integers(0, 24))
        emi = base_inc * rng.uniform(.05, .09)
        new_item("expense", "loan_payment", "Personal/education loan EMI", "monthly", int(rng.integers(1, 11)), t0, emi, rnd=100,
                 t_last=min(47, t0 + int(rng.choice([24, 36])) - 1), jit=0,
                 methods={"Auto Debit": .7, "Net Banking": .3}, merch=None)
    tuition_q = 0.0
    if u["kids"]:
        tuition_q = float(min(base_inc * rng.uniform(.05, .12), 20000))
        anchor = int(rng.choice([0, 1, 2]))     # term-fee month offset within quarter
        new_item("expense", "tuition", "School/tuition term fee", "quarterly", int(rng.integers(3, 15)), anchor, tuition_q, gmean=.08,
                 gsd=.02, cm=4, rnd=100, jit=1, methods={"Net Banking": .35, "UPI": .35, "Bank Transfer": .15, "Cash": .15}, merch=None)

    # ---- savings contributions (transfers)
    sip_p = P["p_sip"] * (.6 if u["income_frequency"] != "monthly" else 1.0)
    if rng.random() < sip_p:
        late = rng.random() < .3
        new_item("transfer", "sip_mutual_fund", SAV_SUBS["sip_mutual_fund"], "monthly", int(rng.integers(1, 11)),
                 int(rng.integers(1, 36)) if late else 0, round_amt(max(500, base_inc * u["save_frac"]), 500), gmean=.07,
                 gsd=.05, cm=int(rng.integers(1, 13)), p_change=.5, rnd=500, jit=0, methods={"Auto Debit": .75, "Net Banking": .25})
    if rng.random() < .12:
        new_item("transfer", "recurring_deposit", SAV_SUBS["recurring_deposit"], "monthly", int(rng.integers(1, 11)), 0,
                 round_amt(max(500, base_inc * rng.uniform(.03, .07)), 500), rnd=500, jit=0,
                 methods={"Auto Debit": .6, "Net Banking": .4})
    if et == "Salaried" and rng.random() < .08:
        new_item("transfer", "ppf_contribution", SAV_SUBS["ppf_contribution"], "annual", 20, 2,
                 float(round(rng.uniform(20000, 150000) / 5000) * 5000), rnd=5000, jit=0, methods={"Net Banking": 1.0})

    # ---- materialise recurring items into rows + recurring file rows
    rec_rows = []
    sc = itertools.count(1)
    for it in items:
        # build segments of equal expected amount
        segs, cur = [], None
        for k, (d, a) in enumerate(zip(it["dates"], it["amts"])):
            if cur is None or a != cur["amt"]:
                cur = dict(amt=a, ks=[k])
                segs.append(cur)
            else:
                cur["ks"].append(k)
        for si, sg in enumerate(segs):
            rid = f"REC{u['idx']:04d}_{next(sc):02d}"
            ks = sg["ks"]
            last = si == len(segs) - 1
            ended = not (last and it["ongoing"])
            rec_rows.append(dict(
                recurring_id=rid, user_id=uid, transaction_type=it["typ"], category=SUB_CAT[it["sub"]], subcategory=it["sub"],
                description=it["desc"], expected_amount=sg["amt"], frequency=it["freq"],
                start_date=it["dates"][ks[0]], end_date=(it["dates"][ks[-1]] if ended else pd.NaT),
                expected_day_of_month=it["day"], active_status="ended" if ended else "active", _it=it, _ks=ks))
            for k in ks:
                it.setdefault("rid_of", {})[k] = rid
    for rr in rec_rows:
        it = rr["_it"]
        sub = it["sub"]
        for k in rr["_ks"]:
            t = it["ts"][k]
            exp_d = it["dates"][k]
            amt = it["amts"][k]
            if it["kind"] == "electricity":
                f = elec_f[t] if it["per"] == 1 else np.mean([elec_f[t], elec_f[max(t - 1, 0)]])
                amt = round(amt * f * float(rng.lognormal(0, .08)), 2)
            elif it["kind"] == "water":
                amt = round(amt * float(rng.lognormal(0, .06)), 2)
            elif it["kind"] == "gas":
                amt = round(amt * float(rng.lognormal(0, .03)), 2)
            day = exp_d.day
            if it["jit"]:
                day = int(min(max(day + int(rng.integers(-2, 3)), 1), 28))
            d = pd.Timestamp(exp_d.year, exp_d.month, day)
            if it["typ"] == "income":
                tm = rand_time(rng, (9, 11)) if sub == "salary" else rand_time(rng, (10, 18))
            elif it["typ"] == "transfer" or (it["methods"] and "Auto Debit" in it["methods"] and
                                             it["methods"].get("Auto Debit", 0) >= .5):
                tm = rand_time(rng, (1, 5))
            else:
                tm = rand_time(rng, (8, 21))
            m = pick_method(sub, amt, allowed=it["methods"]) if it["methods"] else "UPI"
            mch = it["merch"]
            if it["switch"] is not None and d >= it["switch"]:
                mch = it["merch2"]
            if mch is None:
                mch = merchant(sub)
            if it["typ"] == "income":
                emit(d, "income", amt, sub, it["desc"], m, mch, it["mcat"] or "Employer", rid=rr["recurring_id"], freq=it["freq"],
                     time=tm, channel="payroll_credit" if sub == "salary" else None,
                     source="payroll_system" if sub == "salary" else None)
            elif it["typ"] == "transfer":
                emit(d, "transfer", amt, sub, it["desc"], m, mch, SAV_MCAT[sub], rid=rr["recurring_id"], freq=it["freq"], time=tm)
            else:
                cat, mc, nd = REC_SUB[sub]
                emit(d, "expense", amt, sub, it["desc"], m, mch, it["mcat"] or mc, rid=rr["recurring_id"], freq=it["freq"],
                     planned="planned", time=tm)
    recurring_out = [{k: v for k, v in rr.items() if not k.startswith("_")} for rr in rec_rows]

    # fixed monthly-equivalent spending (2022 prices) for sizing the variable pool
    fixed_equiv = sum(it["base"] / it["per"] for it in items if it["typ"] == "expense" and it["t0"] <= 12)
    pool = max(base_inc * u["ratio"] - fixed_equiv, .2 * base_inc)

    # ------------------------------------------------------------ variable spending
    w = {}
    for sub, a in V.items():
        wt = a["w"] * u["taste"][sub]
        if sub in ("fuel", "vehicle_maintenance") and not u["vehicle"]:
            wt = 0
        if sub == "public_transport":
            wt *= .3 if u["vehicle"] else 1.3
        if sub == "taxi":
            wt *= .6 if u["vehicle"] else 1.2
        if sub in ("groceries", "toiletries"):
            wt *= .7 + .15 * u["hh"]
        if sub in ("medicine", "consultation", "diagnostic_tests"):
            wt *= float(np.clip(.5 + (u["age"] - 20) / 40, .5, 2.0))
        if sub in ("books", "online_courses"):
            wt *= 5 if et == "Student" else (1.6 if u["age"] < 30 else 1.0)
            if sub == "books" and u["kids"]:
                wt *= 1.8
        if sub == "food_delivery":
            wt *= u["delivery"]
        if sub == "restaurants":
            wt *= u["foodie"]
        w[sub] = wt
    sw_ = sum(w.values())
    T = {s: pool * w[s] / sw_ for s in V}

    cum = np.cumsum(inc_exp)
    trail = np.array([cum[t] - (cum[t - 12] if t >= 12 else 0) for t in range(N_MONTHS)]) / np.minimum(np.arange(N_MONTHS) + 1, 12)
    rel = trail / inc_exp[0]
    sig = u["sigma"]
    z = rng.standard_normal(N_MONTHS)
    e = np.zeros(N_MONTHS)
    e[0] = sig * z[0]
    for t in range(1, N_MONTHS):
        e[t] = .5 * e[t - 1] + np.sqrt(1 - .25) * sig * z[t]
    f_disc = np.exp(e - sig ** 2 / 2)
    f_ess = np.exp(.5 * e - (.5 * sig) ** 2 / 2)
    if u["profile"] == "inconsistent":
        spike = rng.random(N_MONTHS) < .10
        f_disc[spike] *= rng.uniform(1.3, 1.7, size=int(spike.sum()))
    shift_arr = np.ones(N_MONTHS)
    if u["shift_t"] is not None:
        shift_arr[u["shift_t"]:] = u["shift_mult"]
    tt = np.arange(N_MONTHS) / 12.0
    Evar, Esm = {}, {}
    cap = 3.0 * base_inc
    for sub, a in V.items():
        if T[sub] < 30:
            Evar[sub] = np.zeros(N_MONTHS)
            Esm[sub] = np.zeros(N_MONTHS)
            continue
        infl = (1 + a["infl"]) ** tt
        seas = seasonal_factor(a["season"], state)
        life = np.clip(rel, .7, 1.7) ** a["life"]
        shf = shift_arr if not a["needs"] else np.ones(N_MONTHS)
        base_m = T[sub] * infl * life * shf
        Esm[sub] = base_m
        Evar[sub] = base_m * seas
        shock = f_ess if a["needs"] else f_disc
        lam0 = a["lam"] * COUNT_SCALE * u["freq_mult"]
        for t in range(N_MONTHS):
            lam_t = lam0 * seas[t] ** .5
            n = sample_count(rng, lam_t, u["stability"])
            if n == 0:
                continue
            mu = Evar[sub][t] * shock[t] / lam_t
            sg = a["sig"] * u["amt_disp"]
            amts = rng.lognormal(np.log(max(mu, 1.0)) - .5 * sg ** 2, sg, size=n)
            for amt in amts:
                amt = float(min(max(round(amt), 10), cap))
                d = pd.Timestamp(int(M_YEAR[t]), int(M_MON[t]), pick_day(rng, t, a["weekend"]))
                m = pick_method(sub, amt)
                unpl = "unplanned" if rng.random() < min(a["unpl"] * u["unpl"], .95) else "planned"
                city, st = None, None
                if a["cat"] == "Travel" and rng.random() < .7:
                    c2 = CITIES[int(rng.integers(len(CITIES)))]
                    city, st = c2[0], c2[1]
                emit(d, "expense", amt, sub, a["desc"], m, merchant(sub), a["mcat"], planned=unpl,
                     time=rand_time(rng, a["hrs"]), city=city, st=st)

    # tuition annual admission/charges (June), planned academic-period expense
    if u["kids"]:
        for y in (2022, 2023, 2024, 2025):
            if rng.random() < .8:
                t = (y - 2022) * 12 + 5
                amt = round_amt(tuition_q * rng.uniform(1.0, 1.6) * 1.08 ** (y - 2022), 100)
                d = pd.Timestamp(y, 6, int(rng.integers(3, 20)))
                emit(d, "expense", amt, "tuition", "Annual admission & development fee", pick_method("tuition", amt,
                     allowed={"Net Banking": .4, "UPI": .35, "Bank Transfer": .15, "Cash": .1}), merchant("tuition", REC_MERCH["tuition"]),
                     "School / College", planned="planned", time=rand_time(rng, (9, 17)))

    # incentives / bonuses
    if et == "Salaried":
        sal = salary_item["amt_at"]
        if rng.random() < .45:
            for y, ds in enumerate(FESTIVALS["Diwali"]):
                d = pd.Timestamp(ds) - pd.Timedelta(days=int(rng.integers(1, 7)))
                emit(d, "income", round_amt(sal(d) * rng.uniform(.3, 1.0), 100), "incentives", "Festival bonus", "Bank Transfer",
                     salary_item["merch"], "Employer", time=rand_time(rng, (9, 12)), channel="payroll_credit", source="payroll_system")
        if rng.random() < .3:
            for y in (2022, 2023, 2024, 2025):
                d = pd.Timestamp(y, 4, int(rng.integers(5, 26)))
                emit(d, "income", round_amt(sal(d) * rng.uniform(.5, 2.0), 100), "incentives", "Annual performance bonus", "Bank Transfer",
                     salary_item["merch"], "Employer", time=rand_time(rng, (9, 12)), channel="payroll_credit", source="payroll_system")
        if u["occupation"] in ("Sales Executive", "Marketing Executive"):
            for y in (2022, 2023, 2024, 2025):
                for mo in (3, 6, 9, 12):
                    if rng.random() < .85:
                        d = pd.Timestamp(y, mo, int(rng.integers(10, 21)))
                        emit(d, "income", round_amt(sal(d) * rng.uniform(.05, .3), 100), "incentives", "Quarterly sales incentive",
                             "Bank Transfer", salary_item["merch"], "Employer", time=rand_time(rng, (9, 12)),
                             channel="payroll_credit", source="payroll_system")
    # irregular income
    if et in ("Freelancer", "Business Owner", "Gig/Part-time"):
        sigma_inc = {"Freelancer": .45, "Business Owner": .28, "Gig/Part-time": .18}[et]
        dry = {"Freelancer": .07, "Business Owner": .02, "Gig/Part-time": .02}[et]
        sub = {"Freelancer": "freelance_work", "Business Owner": "business_income", "Gig/Part-time": "freelance_work"}[et]
        clients = ["Client-{}".format(int(rng.integers(100, 999))) for _ in range(4)]
        for t in range(N_MONTHS):
            tot = inc_exp[t]
            if retainer is not None:
                tot = max(tot - retainer["amt_at"](ms[t]), .3 * tot)
            if rng.random() < dry:
                continue
            tot *= float(rng.lognormal(-.5 * sigma_inc ** 2, sigma_inc))
            nk = {"Freelancer": min(1 + int(rng.poisson(1.0)), 4), "Business Owner": min(2 + int(rng.poisson(1.0)), 5),
                  "Gig/Part-time": 2}[et]
            ws = rng.dirichlet(np.ones(nk) * 3)
            for j in range(nk):
                a_ = max(round(tot * ws[j]), 100)
                if et == "Gig/Part-time":
                    day = min(7 + 14 * j + int(rng.integers(-1, 2)), DAYS_IN[t])
                    mch, desc = "Gig Platform Payout (synthetic)", "Platform payout"
                elif et == "Freelancer":
                    day = int(rng.integers(1, DAYS_IN[t] + 1))
                    mch, desc = clients[int(rng.integers(4))], "Project payment"
                else:
                    day = int(rng.integers(1, DAYS_IN[t] + 1))
                    mch, desc = "Customer Receipts" if rng.random() < .6 else "Wholesale Buyer-{}".format(int(rng.integers(10, 99))), "Business receipts"
                m = pick_method(sub, a_, allowed={"Bank Transfer": .5, "UPI": .4, "Cash": (u["cash_pref"] if et == "Business Owner" else .02)})
                emit(pd.Timestamp(int(M_YEAR[t]), int(M_MON[t]), day), "income", float(a_), sub, desc, m, mch, "Client / Customer",
                     time=rand_time(rng, (10, 18)))
    if u["other_income"]:
        for t in range(N_MONTHS):
            if rng.random() < .04:
                desc = str(rng.choice(["Cashback and rewards", "Family gift received", "Tax refund", "Interest credit", "Resale of used item"]))
                emit(pd.Timestamp(int(M_YEAR[t]), int(M_MON[t]), int(rng.integers(1, DAYS_IN[t] + 1))), "income",
                     float(round(rng.uniform(500, 15000))), "other_income", desc,
                     "UPI" if rng.random() < .5 else "Bank Transfer", "Various (synthetic)", "Miscellaneous Credit",
                     time=rand_time(rng, (10, 20)))

    # unexpected expenses, fees, interest
    p_shock = .05 if u["profile"] == "inconsistent" else .025
    shock_subs = ["consultation", "diagnostic_tests", "home_repairs", "electronics", "flights"] + (["vehicle_maintenance"] if u["vehicle"] else [])
    for t in range(N_MONTHS):
        if rng.random() < p_shock:
            sub = str(rng.choice(shock_subs))
            a = V[sub]
            amt = float(min(max(round(pool * rng.uniform(.4, 1.6) * (1.05 ** (t / 12))), 200), cap))
            emit(pd.Timestamp(int(M_YEAR[t]), int(M_MON[t]), int(rng.integers(1, DAYS_IN[t] + 1))), "expense", amt, sub,
                 "Unexpected: " + a["desc"].lower(), pick_method(sub, amt), merchant(sub), a["mcat"], planned="unplanned",
                 time=rand_time(rng, a["hrs"]))
        if rng.random() < .05:
            amt = float(rng.choice([118, 177, 236, 295, 354, 590]))
            emit(pd.Timestamp(int(M_YEAR[t]), int(M_MON[t]), int(rng.integers(1, 29))), "expense", amt, "bank_fees",
                 "Bank service charges", "Auto Debit", merchant("bank_fees"), "Bank", planned="planned", time=rand_time(rng, (1, 5)))
        if u["has_cc"] and u["profile"] in ("overspender", "inconsistent") and rng.random() < .12:
            amt = float(round(rng.uniform(200, max(400, .06 * base_inc))))
            emit(pd.Timestamp(int(M_YEAR[t]), int(M_MON[t]), int(rng.integers(1, 29))), "expense", amt, "interest_charges",
                 "Credit card interest and late charges", "Credit Card", merchant("interest_charges"), "Bank",
                 planned="unplanned", time=rand_time(rng, (1, 5)))

    # lump savings and redemptions (transfers)
    if rng.random() < .3:
        for _ in range(1 + int(rng.random() < .4)):
            t = int(rng.integers(3, 44))
            amt = round_amt(base_inc * rng.uniform(.5, 2.5), 1000)
            emit(pd.Timestamp(int(M_YEAR[t]), int(M_MON[t]), int(rng.integers(1, 29))), "transfer", amt, "fixed_deposit",
                 SAV_SUBS["fixed_deposit"], "Net Banking", merchant("fixed_deposit"), SAV_MCAT["fixed_deposit"], time=rand_time(rng, (9, 17)))
            if rng.random() < .5:
                t2 = t + int(rng.integers(10, 25))
                if t2 < N_MONTHS:
                    emit(pd.Timestamp(int(M_YEAR[t2]), int(M_MON[t2]), int(rng.integers(1, 29))), "transfer",
                         round(amt * float(rng.uniform(1.03, 1.08)), 2), "savings_withdrawal", SAV_SUBS["savings_withdrawal"],
                         "Net Banking", merchant("savings_withdrawal"), SAV_MCAT["savings_withdrawal"], time=rand_time(rng, (9, 17)))

    # own-account transfers and ATM withdrawals (never income / expense)
    if rng.random() < .4:
        for t in range(N_MONTHS):
            if rng.random() < .10:
                emit(pd.Timestamp(int(M_YEAR[t]), int(M_MON[t]), int(rng.integers(1, 29))), "transfer",
                     float(round(rng.uniform(2000, 30000), -2)), "own_account_transfer", TRF_SUBS["own_account_transfer"],
                     "Net Banking", "Own account", "Own Account", time=rand_time(rng, (9, 20)))
    if u["atm_logged"]:
        for t in range(N_MONTHS):
            if cash_tot[t] >= 1500:
                emit(pd.Timestamp(int(M_YEAR[t]), int(M_MON[t]), int(rng.integers(1, 5))), "transfer",
                     float(np.ceil(cash_tot[t] / 500) * 500), "atm_cash_withdrawal", TRF_SUBS["atm_cash_withdrawal"], "Debit Card",
                     "ATM (synthetic)", "ATM", time=rand_time(rng, (8, 21)), channel="atm")

    # ------------------------------------------------------------ failed attempts and refunds
    base_rows = list(rows)
    REFUND_P = {"electronics": .06, "clothing": .08, "household_goods": .04, "food_delivery": .03, "taxi": .02, "flights": .05,
                "hotel_stay": .04, "books": .02, "online_courses": .02}
    for r_ in base_rows:
        if r_["transaction_type"] != "expense" or r_["status"] != "completed":
            continue
        if r_["payment_method"] in ("UPI", "Net Banking", "Credit Card", "Debit Card", "Auto Debit") and rng.random() < .007:
            tw = dict(r_)
            tw["key"] = next(keygen)
            tw["status"] = "failed"
            if r_["recurring_id"] and r_["date"] > START:
                tw["date"] = r_["date"] - pd.Timedelta(days=1)
                tw["time"] = rand_time(rng, (1, 5))
            elif r_["time"]:
                hh_, mm_, ss_ = map(int, r_["time"].split(":"))
                secs = max(hh_ * 3600 + mm_ * 60 + ss_ - int(rng.integers(60, 1200)), 0)
                tw["time"] = f"{secs // 3600:02d}:{(secs % 3600) // 60:02d}:{secs % 60:02d}"
            rows.append(tw)
        p = REFUND_P.get(r_["subcategory"], 0)
        if p and r_["payment_method"] != "Cash" and rng.random() < p:
            lag = int(rng.integers(1, 6)) if r_["subcategory"] in ("taxi", "food_delivery") else int(rng.integers(2, 15))
            rd = r_["date"] + pd.Timedelta(days=lag)
            if rd <= END:
                frac = 1.0 if rng.random() > .35 else float(rng.uniform(.2, .9))
                k = next(keygen)
                rows.append(dict(r_, key=k, date=rd, time=rand_time(rng, (9, 19)) if r_["time"] else "", transaction_type="refund",
                                 amount=round(r_["amount"] * frac, 2), description="Refund: " + r_["description"], is_recurring=0,
                                 recurring_frequency="", recurring_id="", planned="not_applicable", related_key=r_["key"],
                                 status="completed"))

    # ------------------------------------------------------------ budgets
    cat_sm = {c: np.zeros(N_MONTHS) for c in EXP_CATS}
    cat_full = {c: np.zeros(N_MONTHS) for c in EXP_CATS}
    for sub in V:
        cat_sm[V[sub]["cat"]] += Esm[sub]
        cat_full[V[sub]["cat"]] += Evar[sub]
    for it in items:
        if it["typ"] != "expense":
            continue
        cat = SUB_CAT[it["sub"]]
        for t in range(it["t0"], it["t_last"] + 1):
            cat_sm[cat][t] += it["amt_at"](ms[t]) / it["per"]
        for k, t in enumerate(it["ts"]):
            a = it["amts"][k]
            if it["kind"] == "electricity":
                a *= elec_f[t] if it["per"] == 1 else np.mean([elec_f[t], elec_f[max(t - 1, 0)]])
            cat_full[cat][t] += a
    styles = ["fixed_monthly", "seasonal_adjusted", "income_percentage"]
    style = styles[int(rng.choice(3, p=[.45, .35, .20]))]
    essential_cats = {"Housing", "Food", "Transportation", "Utilities", "Healthcare"}
    for cat in EXP_CATS:
        if cat_sm[cat].mean() < 150:
            continue
        pt = .95 if cat in essential_cats else (.85 if cat == "Financial Expenses" else .75)
        if rng.random() > pt:
            continue
        cbias = u["bias"] * float(rng.lognormal(0, .06))
        for t in range(N_MONTHS):
            if style == "fixed_monthly":
                jan = (t // 12) * 12
                b = cat_sm[cat][jan]
            elif style == "seasonal_adjusted":
                b = cat_sm[cat][t] + .8 * (cat_full[cat][t] - cat_sm[cat][t])
            else:
                b = cat_sm[cat][0] * inc_exp[t] / inc_exp[0]
            b = b * cbias * (1 + float(rng.uniform(-.02, .02)) if style != "fixed_monthly" else 1)
            b = max(round(b / (100 if b >= 2000 else 50)) * (100 if b >= 2000 else 50), 200)
            if style == "fixed_monthly":
                created = pd.Timestamp(2022, 1, 1) if t < 12 else pd.Timestamp(int(M_YEAR[t]) - 1, 12, int(rng.integers(20, 32)))
                if t >= 12 and t % 12 != 0:
                    created = budgets[-1]["budget_created_date"]
            else:
                created = max(START, ms[t] - pd.Timedelta(days=int(rng.integers(1, 7))))
            budgets.append(dict(user_id=uid, year=int(M_YEAR[t]), month=int(M_MON[t]), category=cat, budget_amount=float(b),
                                currency="INR", budget_created_date=created, budget_type=style,
                                rollover_enabled=int(u["idx"] % 3 == 0 or rng.random() < .1)))
    return rows, recurring_out, budgets, inc_exp


# ----------------------------------------------------------------------------
# Assembly, summaries, forecasting table
# ----------------------------------------------------------------------------
TX_ORDER = ["transaction_id", "user_id", "transaction_date", "transaction_time", "transaction_type", "amount", "currency",
            "category", "subcategory", "description", "payment_method", "merchant_name", "merchant_category", "account_type",
            "is_recurring", "recurring_frequency", "is_essential", "planned_or_unplanned", "payment_status",
            "transaction_channel", "city", "state", "financial_year", "month", "month_name", "quarter", "day_of_week",
            "weekend_flag", "needs_or_wants", "data_source", "recurring_id", "related_transaction_id"]
RENAME = {"date": "transaction_date", "time": "transaction_time", "essential": "is_essential",
          "planned": "planned_or_unplanned", "status": "payment_status", "channel": "transaction_channel",
          "needs": "needs_or_wants", "source": "data_source"}


def assemble_transactions(all_rows):
    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df["related_key"] = pd.to_numeric(df["related_key"])
    for c in ("merchant_name", "recurring_frequency", "recurring_id", "time"):
        df[c] = df[c].fillna("")
    df = df.sort_values("key").reset_index(drop=True)
    # UPI per-transaction cap (INR 1 lakh): larger receipts/payments are re-labelled as bank transfers
    m = (df.payment_method == "UPI") & (df.amount > 100000)
    n_upi_fix = int(m.sum())
    df.loc[m, "payment_method"] = "Bank Transfer"
    df.loc[m, "source"] = "bank_statement_import"
    df.loc[m, "channel"] = "bank_portal"
    # exact-duplicate detection (all columns except the internal key) and removal; refunds are re-pointed
    ncols = [c for c in df.columns if c != "key"]
    dup_removed, remap = 0, {}
    for _ in range(3):
        first = df.groupby(ncols, dropna=False, sort=False)["key"].transform("first")
        drop = df["key"] != first
        if not drop.any():
            break
        remap.update(dict(zip(df.loc[drop, "key"], first[drop])))
        dup_removed += int(drop.sum())
        df = df[~drop].copy()
        rk = df["related_key"]
        df["related_key"] = rk.map(remap).fillna(rk)
    df = df.sort_values(["date", "time", "key"]).reset_index(drop=True)
    df["transaction_id"] = ["TXN%07d" % (i + 1) for i in range(len(df))]
    idmap = dict(zip(df["key"], df["transaction_id"]))
    df["related_transaction_id"] = df["related_key"].map(idmap).fillna("")
    df = df.rename(columns=RENAME)
    d = df["transaction_date"]
    df["currency"] = "INR"
    df["month"] = d.dt.month
    df["month_name"] = d.dt.month_name()
    df["quarter"] = "Q" + d.dt.quarter.astype(str)
    df["day_of_week"] = d.dt.day_name()
    df["weekend_flag"] = (d.dt.dayofweek >= 5).astype(int)
    fy0 = np.where(d.dt.month >= 4, d.dt.year, d.dt.year - 1)
    df["financial_year"] = ["FY%d-%02d" % (y, (y + 1) % 100) for y in fy0]
    df["transaction_date"] = d.dt.strftime("%Y-%m-%d")
    df["amount"] = df["amount"].round(2)
    return df[TX_ORDER], dict(dup_removed=dup_removed, upi_relabelled=n_upi_fix)


def build_summary(tx, users, bud):
    c = tx[tx.payment_status == "completed"].copy()
    c["year"] = c.transaction_date.dt.year
    c["mon"] = c.transaction_date.dt.month
    key = ["user_id", "year", "mon"]
    idx = pd.MultiIndex.from_product([users.user_id, range(2022, 2026), range(1, 13)], names=key)

    def agg(mask):
        return c[mask].groupby(key)["amount"].sum().reindex(idx, fill_value=0.0)
    ty = c.transaction_type
    inc = agg(ty == "income")
    exp = agg(ty == "expense")
    ess = agg((ty == "expense") & (c.is_essential == 1))
    ref = agg(ty == "refund")
    sv = (ty == "transfer") & (c.category == "Savings and Investments")
    sav_c = agg(sv & (c.subcategory != "savings_withdrawal"))
    sav_r = agg(sv & (c.subcategory == "savings_withdrawal"))
    ncount = c.groupby(key).size().reindex(idx, fill_value=0)
    ecount = c[ty == "expense"].groupby(key).size().reindex(idx, fill_value=0)
    ce = c[ty.isin(["expense", "refund"])].copy()
    ce["signed"] = np.where(ce.transaction_type == "expense", ce.amount, -ce.amount)
    catnet = (ce.groupby(key + ["category"])["signed"].sum().unstack("category")
              .reindex(idx).reindex(columns=EXP_CATS).fillna(0.0))
    bw = (bud.rename(columns={"month": "mon"}).pivot_table(index=key, columns="category", values="budget_amount",
                                                          aggfunc="sum").reindex(idx).reindex(columns=EXP_CATS))
    has = bw.notna()
    total_budget = bw.sum(axis=1)
    actual_b = catnet.where(has, 0.0).sum(axis=1)
    over = ((catnet > bw) & has).sum(axis=1)
    lam = catnet.max(axis=1)
    largest = catnet.idxmax(axis=1).where(lam > 0, "")
    S = pd.DataFrame(index=idx)
    S["total_income"] = inc
    S["total_expenses"] = exp
    S["essential_expenses"] = ess
    S["discretionary_expenses"] = exp - ess
    S["refunds_received"] = ref
    S["net_expenses"] = exp - ref
    S["net_cash_flow"] = inc - S["net_expenses"]
    S["savings_amount"] = sav_c - sav_r
    S["savings_rate"] = np.where(inc > 0, S["savings_amount"] / inc.where(inc > 0, 1), np.nan)
    S["average_transaction_amount"] = np.where(ecount > 0, exp / ecount.where(ecount > 0, 1), np.nan)
    S["transaction_count"] = ncount
    S["largest_expense_category"] = largest
    S["largest_expense_category_amount"] = lam.where(lam > 0, 0.0)
    S["total_budget"] = total_budget
    S["total_actual_budgeted_spending"] = actual_b
    S["budget_utilization_percentage"] = np.where(total_budget > 0, 100 * actual_b / total_budget.where(total_budget > 0, 1), np.nan)
    S["categories_over_budget"] = over
    opening = users.set_index("user_id")["opening_balance_estimate"]
    flow = (S["total_income"] - S["net_expenses"] - S["savings_amount"])
    S["month_end_balance"] = flow.groupby(level=0).cumsum() + opening.reindex(S.index.get_level_values(0)).values
    S = S.reset_index().rename(columns={"mon": "month"})
    for col in S.columns:
        if S[col].dtype == float:
            S[col] = S[col].round(4 if col == "savings_rate" else 2)
    cn = catnet.reset_index().rename(columns={"mon": "month"})
    return S, cn


def festive_flags(state):
    return festival_index(state)


def build_forecast(S, cn, users, rec):
    g = S.merge(cn, on=["user_id", "year", "month"]).merge(
        users[["user_id", "age", "household_size", "monthly_income_estimate", "employment_type", "state"]], on="user_id")
    g["t"] = (g.year - 2022) * 12 + g.month - 1
    g = g.sort_values(["user_id", "t"]).reset_index(drop=True)
    gb = g.groupby("user_id")
    F = pd.DataFrame({"user_id": g.user_id, "year": g.year, "month": g.month})
    F["split"] = ""
    for col in ("age", "household_size", "monthly_income_estimate", "employment_type"):
        F[col] = g[col]
    ne = g["net_expenses"]
    F["exp_m0"] = ne
    for k in (1, 2, 3):
        F[f"exp_lag{k}"] = gb["net_expenses"].shift(k)
    F["exp_target_month_last_year"] = gb["net_expenses"].shift(11)
    roll = lambda col, n, fn="mean", mp=1: gb[col].transform(lambda s: getattr(s.rolling(n, min_periods=mp), fn)())
    F["exp_roll_mean_3"] = roll("net_expenses", 3)
    F["exp_roll_mean_6"] = roll("net_expenses", 6)
    F["exp_roll_mean_12"] = roll("net_expenses", 12)
    F["exp_roll_std_6"] = roll("net_expenses", 6, "std", 2)
    F["history_months"] = g["t"] + 1
    F["essential_exp_m0"] = g["essential_expenses"]
    F["discretionary_exp_m0"] = g["discretionary_expenses"]
    F["income_m0"] = g["total_income"]
    F["income_roll_mean_3"] = roll("total_income", 3)
    F["income_roll_mean_12"] = roll("total_income", 12)
    F["savings_m0"] = g["savings_amount"]
    F["txn_count_m0"] = g["transaction_count"]
    F["avg_txn_amount_m0"] = g["average_transaction_amount"]
    for cat in EXP_CATS:
        F[f"cat_{slug(cat)}_m0"] = g[cat]
    for cat in EXP_CATS:
        F[f"cat_{slug(cat)}_roll3"] = g.groupby("user_id")[cat].transform(lambda s: s.rolling(3, min_periods=1).mean())
    F["budget_total_m0"] = g["total_budget"]
    F["budget_total_target"] = gb["total_budget"].shift(-1)
    F["budget_utilization_m0"] = g["budget_utilization_percentage"]
    F["categories_over_budget_m0"] = g["categories_over_budget"]
    # scheduled recurring expenses (known in advance from recurring_transactions.csv) due in the target month
    due = defaultdict(float)
    for r in rec[rec.transaction_type == "expense"].itertuples():
        s = (r.start_date.year - 2022) * 12 + r.start_date.month - 1
        e = 47 if pd.isna(r.end_date) else (r.end_date.year - 2022) * 12 + r.end_date.month - 1
        for t in range(s, e + 1, PERIOD[r.frequency]):
            due[(r.user_id, t)] += r.expected_amount
    F["expected_recurring_target"] = [round(due.get((u, t + 1), 0.0), 2) for u, t in zip(g.user_id, g.t)]
    tm = pd.Series((g["month"].values % 12) + 1, index=g.index)          # calendar month of the target month
    F["target_year"] = np.where(g["month"].values == 12, g["year"].values + 1, g["year"].values)
    F["target_month"] = tm
    F["target_month_sin"] = np.round(np.sin(2 * np.pi * tm / 12), 6)
    F["target_month_cos"] = np.round(np.cos(2 * np.pi * tm / 12), 6)
    fidx = {s: festival_index(s) for s in users.state.unique()}
    F["target_festival_index"] = [round(float(fidx[s][min(t + 1, 47)]), 4) for s, t in zip(g.state, g.t)]
    F["target_is_festive_peak"] = (F["target_festival_index"] >= 0.5).astype(int)
    F["target_is_summer_travel"] = tm.isin([4, 5, 6]).astype(int)
    F["target_is_year_end"] = (tm == 12).astype(int)
    F["target_is_academic_fee_month"] = tm.isin([6, 7]).astype(int)
    F["target_next_month_expenses"] = gb["net_expenses"].shift(-1)
    F = F[F.target_next_month_expenses.notna()].copy()
    F["split"] = np.where(F.target_year <= 2023, "train", np.where(F.target_year == 2024, "validation", "test"))
    for col in F.columns:
        if F[col].dtype == float:
            F[col] = F[col].round(2) if "sin" not in col and "cos" not in col and "index" not in col else F[col]
    return F.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Forecasting experiment (time-aware split, no shuffling)
# ----------------------------------------------------------------------------
def run_ml(fc):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    drop = ["user_id", "year", "month", "split", "target_year", "target_next_month_expenses"]
    X = pd.get_dummies(fc.drop(columns=drop), columns=["employment_type"], dtype=float)
    y = fc["target_next_month_expenses"].values
    sp = fc["split"].values

    def met(yt, p):
        mae = mean_absolute_error(yt, p)
        rmse = float(np.sqrt(mean_squared_error(yt, p)))
        mk = yt >= 1000
        mape = float(np.mean(np.abs(yt[mk] - p[mk]) / yt[mk]) * 100) if mk.sum() >= 30 else float("nan")
        return mae, rmse, mape
    lines = []
    lines.append("Features: %d columns (numeric + one-hot employment_type). Split by TARGET month: train = targets in 2022-2023 "
                 "(%d rows), validation = 2024 (%d), test = 2025 (%d). No shuffling." %
                 (X.shape[1], (sp == "train").sum(), (sp == "validation").sum(), (sp == "test").sum()))
    lines.append("MAPE is computed only on rows with target >= INR 1,000 (otherwise it is unstable). Models predict next-month NET expenses "
                 "(expenses - linked refunds; transfers excluded).")
    results = {}

    def models():
        return {"LinearRegression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression()),
                "RandomForest": make_pipeline(SimpleImputer(strategy="median"),
                                              RandomForestRegressor(n_estimators=200, max_depth=14, min_samples_leaf=5,
                                                                    n_jobs=-1, random_state=SEED))}

    def evaluate(train_mask, eval_mask, label):
        yt = y[eval_mask]
        Xe = X[eval_mask]
        rows = []
        for name, col in (("Baseline: last month (naive)", "exp_m0"), ("Baseline: 12-month historical average", "exp_roll_mean_12"),
                          ("Baseline: seasonal naive (same month last year)", "exp_target_month_last_year")):
            p = Xe[col].values.copy()
            ok = ~np.isnan(p)
            rows.append((name, ) + met(yt[ok], p[ok]) + (int(ok.sum()),))
        fitted = {}
        for name, mdl in models().items():
            mdl.fit(X[train_mask], y[train_mask])
            p = np.clip(mdl.predict(Xe), 0, None)
            rows.append((name, ) + met(yt, p) + (len(yt),))
            fitted[name] = mdl
        lines.append("")
        lines.append(label)
        lines.append("%-52s %12s %12s %9s %7s" % ("model", "MAE (INR)", "RMSE (INR)", "MAPE %", "n"))
        for r in rows:
            lines.append("%-52s %12.0f %12.0f %9.1f %7d" % r)
        results[label] = rows
        return fitted
    fitted = evaluate(sp == "train", sp == "validation", "VALIDATION 2024 (models fit on 2022-2023 targets only)")
    evaluate((sp == "train") | (sp == "validation"), sp == "test", "TEST 2025 (models refit on 2022-2024 targets; test never used for fitting)")
    rf = fitted["RandomForest"].steps[-1][1]
    imp = sorted(zip(X.columns, rf.feature_importances_), key=lambda z: -z[1])[:10]
    lines.append("")
    lines.append("Top-10 RandomForest feature importances (fit on train): " + "; ".join("%s=%.3f" % (a, b) for a, b in imp))
    lines.append("Seasonal naive is evaluated on rows where the same month of the previous year exists (all validation/test rows qualify).")
    lines.append("Interpretation: accuracy on synthetic data reflects the generating rules (smooth trends, bounded noise, known recurring bills) "
                 "and says little about performance on real household data, which has regime changes, one-off shocks and missing records.")
    return lines, results


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
INTENTIONAL_MISSING = {
    ("transactions.csv", "transaction_time"): "unknown time-of-day: ~50% of manual cash entries carry no time",
    ("transactions.csv", "merchant_name"): "unknown merchant: informal cash payments (35% of cash expenses)",
    ("transactions.csv", "recurring_frequency"): "not applicable: non-recurring transactions",
    ("transactions.csv", "recurring_id"): "not applicable: non-recurring transactions",
    ("transactions.csv", "related_transaction_id"): "not applicable: only refunds link to an original expense",
    ("recurring_transactions.csv", "end_date"): "not applicable: pattern still active at 2025-12-31",
    ("monthly_financial_summary.csv", "savings_rate"): "undefined: total_income = 0 in that user-month",
    ("monthly_financial_summary.csv", "average_transaction_amount"): "undefined: no completed expenses in that user-month",
    ("monthly_financial_summary.csv", "largest_expense_category"): "none: no net expense in that user-month",
    ("monthly_financial_summary.csv", "budget_utilization_percentage"): "undefined: user has no budget that month",
    ("monthly_expense_forecasting.csv", "exp_lag1"): "insufficient history at the start of 2022",
    ("monthly_expense_forecasting.csv", "exp_lag2"): "insufficient history at the start of 2022",
    ("monthly_expense_forecasting.csv", "exp_lag3"): "insufficient history at the start of 2022",
    ("monthly_expense_forecasting.csv", "exp_target_month_last_year"): "insufficient history (first 11 prediction months)",
    ("monthly_expense_forecasting.csv", "exp_roll_std_6"): "needs >= 2 months of history",
    ("monthly_expense_forecasting.csv", "avg_txn_amount_m0"): "no completed expenses in month",
    ("monthly_expense_forecasting.csv", "budget_utilization_m0"): "no budget in month",
    ("monthly_expense_forecasting.csv", "budget_total_target"): "n/a",
}
FIXED_RECUR_SUBS = None


def validate(out, tx, users, bud, rec, S, fc, prep, ml_lines):
    L, fails = [], []

    def chk(name, bad):
        L.append("  [%s] %s: %s" % ("OK " if bad == 0 else "FAIL", name, bad))
        if bad:
            fails.append(name)
    L.append("DATASET VALIDATION REPORT - AI Personal Finance and Smart Expense Predictor (synthetic)")
    L.append("Generated by generate_dataset.py, seed %d. All data synthetic." % SEED)
    L.append("")
    L.append("1. ROW COUNTS")
    files = {"transactions.csv": tx, "users.csv": users, "monthly_budgets.csv": bud, "recurring_transactions.csv": rec,
             "monthly_financial_summary.csv": S, "monthly_expense_forecasting.csv": fc}
    for k, v in files.items():
        L.append("  %-34s %8d rows x %d cols" % (k, len(v), v.shape[1]))
    L.append("  Unique users in users.csv: %d | in transactions.csv: %d | in summary: %d" %
             (users.user_id.nunique(), tx.user_id.nunique(), S.user_id.nunique()))
    L.append("  Unique transaction_ids: %d (rows %d)" % (tx.transaction_id.nunique(), len(tx)))
    L.append("")
    L.append("2. INTEGRITY CHECKS (count of violations; 0 = pass)")
    chk("duplicate transaction_id", int(tx.transaction_id.duplicated().sum()))
    chk("exact duplicate rows remaining (all columns except id)", int(tx.drop(columns=["transaction_id", "related_transaction_id"]).duplicated().sum()) if False else int(tx.drop(columns=["transaction_id"]).duplicated().sum()))
    L.append("  (info) exact duplicate rows detected and removed before saving: %d; UPI >100k re-labelled as Bank Transfer: %d" %
             (prep["dup_removed"], prep["upi_relabelled"]))
    d = pd.to_datetime(tx.transaction_date, errors="coerce")
    chk("invalid or unparsable dates", int(d.isna().sum()))
    chk("dates outside 2022-01-01..2025-12-31", int(((d < START) | (d > END)).sum()))
    chk("month column disagrees with date", int((tx.month != d.dt.month).sum()))
    chk("month_name disagrees with date", int((tx.month_name != d.dt.month_name()).sum()))
    chk("day_of_week/weekend_flag disagree with date", int(((tx.day_of_week != d.dt.day_name()) | (tx.weekend_flag != (d.dt.dayofweek >= 5).astype(int))).sum()))
    chk("non-positive or non-finite amounts", int((~np.isfinite(tx.amount) | (tx.amount <= 0)).sum()))
    chk("currency != INR", int((tx.currency != "INR").sum()))
    chk("transactions referencing unknown user_id", int((~tx.user_id.isin(users.user_id)).sum()))
    chk("invalid transaction_type", int((~tx.transaction_type.isin(["income", "expense", "refund", "transfer"])).sum()))
    pairs = set(zip(tx.category, tx.subcategory))
    chk("category/subcategory pairs outside taxonomy", len(pairs - VALID_PAIRS))
    chk("expense/refund outside the 12 expense categories", int((tx.transaction_type.isin(["expense", "refund"]) & ~tx.category.isin(EXP_CATS)).sum()))
    chk("income rows outside category 'Income'", int(((tx.transaction_type == "income") & (tx.category != "Income")).sum()))
    chk("transfers outside Savings/Transfers categories (would risk being counted as spend/income)",
        int(((tx.transaction_type == "transfer") & ~tx.category.isin(["Savings and Investments", "Transfers"])).sum()))
    chk("UPI transactions above INR 1,00,000", int(((tx.payment_method == "UPI") & (tx.amount > 100000)).sum()))
    chk("is_recurring=1 without recurring_id/frequency", int(((tx.is_recurring == 1) & (tx.recurring_id.isna() | tx.recurring_frequency.isna())).sum()))
    # refunds
    rf = tx[tx.transaction_type == "refund"].merge(tx[["transaction_id", "user_id", "transaction_type", "subcategory", "amount",
                                                       "transaction_date", "payment_status"]],
                                                   left_on="related_transaction_id", right_on="transaction_id", how="left",
                                                   suffixes=("", "_orig"))
    chk("refunds without a valid linked original", int(rf.transaction_id_orig.isna().sum()))
    chk("refunds linked to non-expense / other user / other subcategory / not completed",
        int(((rf.transaction_type_orig != "expense") | (rf.user_id != rf.user_id_orig) | (rf.subcategory != rf.subcategory_orig) |
             (rf.payment_status_orig != "completed")).sum()))
    chk("refund amount > original or refund dated before original",
        int(((rf.amount > rf.amount_orig + 0.005) | (pd.to_datetime(rf.transaction_date) < pd.to_datetime(rf.transaction_date_orig))).sum()))
    chk("non-refund rows carrying related_transaction_id", int(((tx.transaction_type != "refund") & tx.related_transaction_id.notna()).sum()))
    # budgets
    chk("budgets: unknown user", int((~bud.user_id.isin(users.user_id)).sum()))
    chk("budgets: invalid year/month/category", int((~bud.year.between(2022, 2025) | ~bud.month.between(1, 12) | ~bud.category.isin(EXP_CATS)).sum()))
    chk("budgets: duplicate user-year-month-category", int(bud.duplicated(["user_id", "year", "month", "category"]).sum()))
    chk("budgets: non-positive amounts", int((bud.budget_amount <= 0).sum()))
    chk("budgets: duplicate budget_id", int(bud.budget_id.duplicated().sum()))
    # recurring agreement
    sd, ed = pd.to_datetime(rec.start_date), pd.to_datetime(rec.end_date)
    exp_rows = []
    for r, s_, e_ in zip(rec.itertuples(), sd, ed):
        s = (s_.year - 2022) * 12 + s_.month - 1
        e = 47 if pd.isna(e_) else (e_.year - 2022) * 12 + e_.month - 1
        for t in range(s, e + 1, PERIOD[r.frequency]):
            exp_rows.append((r.recurring_id, t, r.expected_amount, r.subcategory))
    E = pd.DataFrame(exp_rows, columns=["recurring_id", "t", "expected_amount", "subcategory"])
    A = tx[tx.recurring_id.notna() & (tx.payment_status == "completed")].copy()
    A["t"] = (pd.to_datetime(A.transaction_date).dt.year - 2022) * 12 + pd.to_datetime(A.transaction_date).dt.month - 1
    M = E.merge(A[["recurring_id", "t", "amount", "user_id"]], on=["recurring_id", "t"], how="outer", indicator=True)
    chk("recurring: scheduled occurrence missing in transactions", int((M._merge == "left_only").sum()))
    chk("recurring: transaction without scheduled occurrence / duplicate occurrences", int((M._merge == "right_only").sum()) + int(A.duplicated(["recurring_id", "t"]).sum()))
    both = M[M._merge == "both"]
    fixed = ~both.subcategory.isin(["electricity", "water", "gas"])
    chk("recurring: fixed-amount items whose amount != expected_amount", int((np.abs(both[fixed].amount - both[fixed].expected_amount) > 0.01).sum()))
    ratio = both[~fixed].amount / both[~fixed].expected_amount
    chk("recurring: utility bills outside 0.4x-2.5x of expected amount", int(((ratio < .4) | (ratio > 2.5)).sum()))
    chk("recurring: recurring_id unknown / user mismatch", int((~A.recurring_id.isin(rec.recurring_id)).sum()) +
        int((A.merge(rec[["recurring_id", "user_id"]], on="recurring_id", suffixes=("", "_r")).eval("user_id != user_id_r")).sum()))
    # summary reconciliation (independent pivot of the saved transactions)
    c = tx[tx.payment_status == "completed"].copy()
    c["year"] = pd.to_datetime(c.transaction_date).dt.year
    c["month_"] = pd.to_datetime(c.transaction_date).dt.month
    P = c.pivot_table(index=["user_id", "year", "month_"], columns="transaction_type", values="amount", aggfunc="sum", fill_value=0.0)
    Sx = S.set_index(["user_id", "year", "month"])
    P.index.names = ["user_id", "year", "month"]
    P = P.reindex(Sx.index, fill_value=0.0)
    for col, ser in (("total_income", P["income"]), ("total_expenses", P["expense"]), ("refunds_received", P["refund"]),
                     ("net_cash_flow", P["income"] - P["expense"] + P["refund"])):
        chk("reconcile %s (user-month mismatches)" % col, int((np.abs(Sx[col] - ser) > 0.011).sum()))
    chk("reconcile essential+discretionary = expenses", int((np.abs(Sx.essential_expenses + Sx.discretionary_expenses - Sx.total_expenses) > .011).sum()))
    chk("summary: 150 users x 48 months, unique keys", int(len(S) != N_USERS * 48 or S.duplicated(["user_id", "year", "month"]).any()))
    chk("summary: savings_rate defined only where income > 0", int((S.savings_rate.notna() != (S.total_income > 0)).sum()))
    chk("summary: transfers not leaking into income/expense (income from own-account/ATM/savings)",
        int(c[(c.transaction_type.isin(["income", "expense"])) & c.category.isin(["Transfers", "Savings and Investments"])].shape[0]))
    ann_tx = c[c.transaction_type.isin(["income", "expense", "refund"])].groupby(["year", "transaction_type"]).amount.sum().unstack()
    ann_s = S.groupby("year")[["total_income", "total_expenses", "refunds_received"]].sum()
    chk("reconcile annual totals (summary vs transactions)", int((np.abs(ann_s.values - ann_tx[["income", "expense", "refund"]].values) > 0.05).sum()))
    # forecasting leakage / construction checks
    St = S.copy()
    St["t"] = (St.year - 2022) * 12 + St.month - 1
    ser = {u: g_.sort_values("t").net_expenses.values for u, g_ in St.groupby("user_id")}
    rng = np.random.default_rng(SEED)
    samp = fc.sample(min(400, len(fc)), random_state=SEED)
    bad_t = bad_f = bad_l = 0
    for r in samp.itertuples():
        t = (r.year - 2022) * 12 + r.month - 1
        v = ser[r.user_id]
        if abs(v[t + 1] - r.target_next_month_expenses) > 0.011:
            bad_t += 1
        if abs(np.mean(v[max(0, t - 11):t + 1]) - r.exp_roll_mean_12) > 0.02 or abs(v[t] - r.exp_m0) > 0.011:
            bad_f += 1
        if t >= 1 and abs(v[t - 1] - r.exp_lag1) > 0.011:
            bad_l += 1
    chk("forecast target != next month's net expenses (sampled 400 rows)", bad_t)
    chk("forecast features inconsistent with history up to month t (sampled)", bad_f + bad_l)
    allowed_target_cols = {"target_next_month_expenses", "budget_total_target", "expected_recurring_target"}
    leaky = [c_ for c_ in fc.columns if "next" in c_ and c_ not in allowed_target_cols]
    chk("unexpected forward-looking columns in forecast table", len(leaky))
    chk("forecast split ordering (train < validation < test by target month)",
        int(not (fc[fc.split == "train"].target_year.max() < 2024 and fc[fc.split == "validation"].target_year.eq(2024).all()
                 and fc[fc.split == "test"].target_year.eq(2025).all())))
    chk("forecast: duplicate user-year-month rows", int(fc.duplicated(["user_id", "year", "month"]).sum()))
    L.append("")
    L.append("3. MISSING VALUES (blank cells are intentional where documented)")
    for fname, df_ in files.items():
        na = df_.isna().sum()
        na = na[na > 0]
        if na.empty:
            L.append("  %s: none" % fname)
        for col, n in na.items():
            why = INTENTIONAL_MISSING.get((fname, col))
            if why is None:
                fails.append("unexpected missing %s.%s" % (fname, col))
            L.append("  %s.%s: %d (%.2f%%) - %s" % (fname, col, n, 100 * n / len(df_), why or "UNEXPECTED"))
    L.append("")
    L.append("4. DISTRIBUTIONS")
    L.append("  Transaction type:")
    for k, v in tx.transaction_type.value_counts().items():
        L.append("    %-10s %7d" % (k, v))
    L.append("  Payment status: " + ", ".join("%s=%d" % kv for kv in tx.payment_status.value_counts().items()))
    L.append("  Payment method: " + ", ".join("%s=%d" % kv for kv in tx.payment_method.value_counts().items()))
    L.append("  Category (rows | completed expense INR):")
    ce = c[c.transaction_type == "expense"].groupby("category").amount.sum()
    for k, v in tx.category.value_counts().items():
        L.append("    %-26s %7d | %14.0f" % (k, v, ce.get(k, 0.0)))
    L.append("  Monthly transaction counts (all rows):")
    mc = tx.groupby(pd.to_datetime(tx.transaction_date).dt.to_period("M")).size()
    for y in range(2022, 2026):
        L.append("    %d: " % y + " ".join("%5d" % mc.get(pd.Period("%d-%02d" % (y, m)), 0) for m in range(1, 13)))
    L.append("  Annual totals, completed rows, INR (income | gross expenses | refunds | net expenses | net cash flow | net savings contributions):")
    for y, r in S.groupby("year")[["total_income", "total_expenses", "refunds_received", "net_expenses", "net_cash_flow", "savings_amount"]].sum().iterrows():
        L.append("    %d: %15.0f | %15.0f | %12.0f | %15.0f | %15.0f | %13.0f" % (y, *r.values))
    L.append("")
    L.append("5. BEHAVIOUR SANITY CHECKS (by financial_profile)")
    SU = S.merge(users[["user_id", "financial_profile", "employment_type", "monthly_income_estimate"]], on="user_id")
    SU["over"] = (SU.budget_utilization_percentage > 100).astype(float).where(SU.budget_utilization_percentage.notna())
    SU["sr"] = SU.savings_rate
    g1 = SU.groupby("financial_profile").agg(users=("user_id", "nunique"), mean_budget_util=("budget_utilization_percentage", "mean"),
                                             share_months_over_total_budget=("over", "mean"),
                                             median_savings_rate=("sr", "median"),
                                             share_months_negative_cash_flow=("net_cash_flow", lambda s: (s < 0).mean()))
    L.append(g1.round(3).to_string())
    cv = SU[SU.total_income >= 0].groupby(["employment_type", "user_id"]).total_income.agg(lambda s: s.std() / s.mean() if s.mean() > 0 else np.nan).groupby("employment_type").median()
    L.append("  Median per-user income coefficient of variation by employment type: " + ", ".join("%s=%.2f" % kv for kv in cv.items()))
    obs = S.groupby("user_id").total_income.mean().reindex(users.user_id).values / users.monthly_income_estimate.values
    L.append("  Observed mean monthly income / monthly_income_estimate: p10=%.2f median=%.2f p90=%.2f" % tuple(np.nanpercentile(obs, [10, 50, 90])))
    heavy = S.assign(o=(S.categories_over_budget > 0)).groupby("user_id").o.mean()
    L.append("  Users with >=1 category over budget in >=50%% of months: %d of %d" % ((heavy >= .5).sum(), len(heavy)))
    L.append("")
    L.append("6. FORECASTING EXPERIMENT (scikit-learn)")
    L.extend(ml_lines if ml_lines else ["  skipped (--no-ml)"])
    L.append("")
    L.append("7. RESULT: %s" % ("ALL CHECKS PASSED" if not fails else "FAILED CHECKS: " + "; ".join(fails)))
    L.append("")
    L.append("8. KNOWN LIMITATIONS")
    for s in [
        "All values are synthetic and rule-generated; distributions are plausible, not calibrated to real Indian household surveys.",
        "Volume: scheduled recurring items alone produce ~40k rows (150 users x 48 months), so the 30k-50k target cannot hold together with realistic "
        "variable-spend logging. COUNT_SCALE=0.55 keeps entry sizes plausible (each variable entry aggregates roughly one to two purchases/hauls) and yields a row count above 50k. "
        "Lower COUNT_SCALE to shrink the file, at the cost of unrealistically large single entries.",
        "month_end_balance is a tracked liquid-balance estimate (synthetic opening balance + cumulative income - net expenses - net savings contributions). "
        "It is NOT net worth: no asset valuations, credit-card dues, or loan principal are modelled, and it can be negative for overspenders.",
        "Credit-card spend is counted when incurred (no separate card-bill settlement transaction) to avoid double counting.",
        "Festival dates are approximate calendar dates; regional festival mapping is simplified.",
        "Budgets are generated from the user's expected (not realised) spending, so budget values legitimately carry information about spending, as real budgets do.",
        "Failed payments are kept as rows with payment_status='failed' and are excluded from every total.",
        "Forecast accuracy on synthetic data must not be read as real-world accuracy; validate on real, consented data before any deployment.",
        "No fraud/anomaly ground-truth label is provided. See methodology for the documented rule-based unusual-spending screen.",
    ]:
        L.append("  - " + s)
    return L, fails


# ----------------------------------------------------------------------------
# Data dictionary and docs
# ----------------------------------------------------------------------------
def data_dictionary(fc_cols):
    R = []

    def add(f, block):
        for ln in block.strip().splitlines():
            col, dt, req, desc = [p.strip() for p in ln.split("|", 3)]
            R.append(dict(file=f, column=col, dtype=dt, required=req, description=desc))
    add("transactions.csv", """
transaction_id|string|required|Unique stable id TXN#######, assigned after sorting by date, time
user_id|string|required|Anonymous user id U0001..U0150; FK to users.csv
transaction_date|date YYYY-MM-DD|required|Date of the transaction (2022-01-01..2025-12-31)
transaction_time|time HH:MM:SS|optional|Time of day; blank when unknown (some manual cash entries)
transaction_type|string|required|income | expense | refund | transfer
amount|float INR|required|Positive magnitude (2 decimals); transaction_type gives the financial meaning
currency|string|required|Always INR
category|string|required|Top-level taxonomy category (12 expense categories, Income, Savings and Investments, Transfers)
subcategory|string|required|Taxonomy subcategory (see methodology)
description|string|required|Short text description
payment_method|string|required|UPI, Debit Card, Credit Card, Cash, Net Banking, Bank Transfer, Digital Wallet, Auto Debit
merchant_name|string|optional|Fictional merchant/counterparty; blank for informal cash payments
merchant_category|string|required|Generic merchant class
account_type|string|required|savings_account, current_account, credit_card, cash_in_hand, digital_wallet
is_recurring|int 0/1|required|1 when generated from a recurring pattern (recurring_id set)
recurring_frequency|string|optional|monthly, bimonthly, quarterly, annual; blank if not recurring
is_essential|int 0/1|required|1 for needs-type expense/refund rows, else 0
planned_or_unplanned|string|required|planned | unplanned | not_applicable
payment_status|string|required|completed | failed (failed rows are excluded from all totals)
transaction_channel|string|required|online, in_store, cash_counter, bank_portal, auto_debit, payroll_credit, atm
city|string|required|City where transaction occurred (user's city; travel rows may differ)
state|string|required|State of that city
financial_year|string|required|Indian FY (Apr-Mar), e.g. FY2022-23; Jan-Mar 2022 falls in FY2021-22
month|int|required|Calendar month 1-12 derived from transaction_date
month_name|string|required|English month name
quarter|string|required|Calendar quarter Q1-Q4
day_of_week|string|required|Monday..Sunday
weekend_flag|int 0/1|required|1 for Saturday/Sunday
needs_or_wants|string|required|needs | wants | savings | not_applicable
data_source|string|required|upi_app_export, card_statement, manual_entry, bank_statement_import, wallet_export, payroll_system
recurring_id|string|optional|FK to recurring_transactions.csv; blank if not recurring
related_transaction_id|string|optional|For refunds: transaction_id of the original expense; blank otherwise
""")
    add("users.csv", """
user_id|string|required|Anonymous id U0001..U0150
age|int|required|Age in years
occupation|string|required|Synthetic occupation label
employment_type|string|required|Salaried, Freelancer, Business Owner, Gig/Part-time, Student, Retired
household_size|int|required|Persons in household
city|string|required|City of residence
state|string|required|State/UT
monthly_income_estimate|float INR|required|Declared average monthly income over the window (rounded to 500); irregular earners deviate month to month
income_frequency|string|required|monthly, irregular, seasonal
preferred_payment_method|string|required|Self-declared preferred method
financial_profile|string|required|disciplined_saver, balanced, overspender, inconsistent (synthetic behaviour archetype)
profile_created_date|date|required|App profile creation date (Oct-Dec 2021)
opening_balance_estimate|float INR|required|Synthetic opening liquid balance on 2022-01-01 used by month_end_balance
""")
    add("monthly_budgets.csv", """
budget_id|string|required|Unique id BUD######
user_id|string|required|FK to users.csv
year|int|required|2022-2025
month|int|required|1-12
category|string|required|One of the 12 expense categories; categories a user does not budget are absent
budget_amount|float INR|required|Planned monthly limit
currency|string|required|INR
budget_created_date|date|required|Date the budget was set (before or at start of the month)
budget_type|string|required|fixed_monthly (annual reset), seasonal_adjusted, income_percentage
rollover_enabled|int 0/1|required|Whether unused budget rolls over (informational)
""")
    add("recurring_transactions.csv", """
recurring_id|string|required|Unique id REC####_## ; one row per constant-amount segment of a recurring pattern
user_id|string|required|FK to users.csv
transaction_type|string|required|income | expense | transfer (savings contributions)
category|string|required|Taxonomy category
subcategory|string|required|Taxonomy subcategory (extra column)
description|string|required|Pattern description
expected_amount|float INR|required|Expected base amount for the segment; utilities vary around it with seasonality
frequency|string|required|monthly, bimonthly, quarterly, annual
start_date|date|required|First scheduled occurrence in the segment
end_date|date|optional|Last occurrence of an ended segment; blank if active at 2025-12-31
expected_day_of_month|int|required|Expected due day (actual date jitters +-2 days for some patterns)
active_status|string|required|active | ended (ended also when the amount changed and a new segment began)
""")
    add("monthly_financial_summary.csv", """
user_id|string|required|FK to users.csv
year|int|required|2022-2025
month|int|required|1-12
total_income|float INR|required|Sum of completed income transactions
total_expenses|float INR|required|Gross sum of completed expense transactions (refunds not netted)
essential_expenses|float INR|required|Completed expenses with is_essential=1
discretionary_expenses|float INR|required|total_expenses - essential_expenses
refunds_received|float INR|required|Sum of completed refunds
net_cash_flow|float INR|required|total_income - (total_expenses - refunds_received); transfers excluded
savings_amount|float INR|required|Net savings contributions: SIP/RD/PPF/FD transfers minus redemptions; not an expense
savings_rate|float|optional|savings_amount / total_income when total_income > 0 else blank
average_transaction_amount|float INR|optional|total_expenses / completed expense count; blank if none
transaction_count|int|required|Completed transactions of all types
largest_expense_category|string|optional|Expense category with the highest net spend; blank if none
largest_expense_category_amount|float INR|required|Its net spend (0 if none)
total_budget|float INR|required|Sum of the user's category budgets for the month (0 if none)
total_actual_budgeted_spending|float INR|required|Net spend (expenses - refunds) in categories that have a budget that month
budget_utilization_percentage|float|optional|100 * total_actual_budgeted_spending / total_budget; blank if no budget
categories_over_budget|int|required|Count of budgeted categories with net spend > budget
month_end_balance|float INR|required|Tracked liquid balance estimate = opening_balance_estimate + cumulative(income - net expenses - net savings); NOT net worth
net_expenses|float INR|required|total_expenses - refunds_received (the eligible expense measure used for forecasting)
""")
    fd = {
        "user_id": "FK to users.csv", "year": "Year of the prediction (feature) month", "month": "Month of the prediction (feature) month",
        "split": "train (target in 2022-23) | validation (target in 2024) | test (target in 2025)",
        "age": "Static user attribute", "household_size": "Static user attribute", "monthly_income_estimate": "Static user attribute (INR)",
        "employment_type": "Static user attribute (categorical)",
        "exp_m0": "Net expenses in the prediction month (INR), known at prediction date",
        "exp_lag1": "Net expenses 1 month before the prediction month", "exp_lag2": "Net expenses 2 months before",
        "exp_lag3": "Net expenses 3 months before",
        "exp_target_month_last_year": "Net expenses in the same calendar month as the target, one year earlier (seasonal-naive feature)",
        "exp_roll_mean_3": "Mean net expenses over last 3 months incl. prediction month (min 1 month)",
        "exp_roll_mean_6": "Mean over last 6 months", "exp_roll_mean_12": "Mean over last 12 months (expanding when fewer exist)",
        "exp_roll_std_6": "Std dev over last 6 months (>=2 months needed)",
        "history_months": "Months of history available up to and including the prediction month",
        "essential_exp_m0": "Gross essential expenses in prediction month", "discretionary_exp_m0": "Gross discretionary expenses in prediction month",
        "income_m0": "Income in prediction month", "income_roll_mean_3": "Mean income last 3 months", "income_roll_mean_12": "Mean income last 12 months",
        "savings_m0": "Net savings contributions in prediction month", "txn_count_m0": "Completed transaction count in prediction month",
        "avg_txn_amount_m0": "Average expense amount in prediction month",
        "budget_total_m0": "Total budget of the prediction month", "budget_total_target": "Total budget of the target month (budgets are set before the month starts)",
        "budget_utilization_m0": "Budget utilisation % in the prediction month", "categories_over_budget_m0": "Categories over budget in prediction month",
        "expected_recurring_target": "Sum of expected_amount of recurring EXPENSE patterns scheduled in the target month (known in advance)",
        "target_year": "Calendar year of the target month", "target_month": "Calendar month of the target month",
        "target_month_sin": "sin(2*pi*target_month/12)", "target_month_cos": "cos(2*pi*target_month/12)",
        "target_festival_index": "0-1 festival-shopping intensity of the target month for the user's state (public calendar knowledge)",
        "target_is_festive_peak": "1 if target_festival_index >= 0.5", "target_is_summer_travel": "1 if target month is Apr-Jun",
        "target_is_year_end": "1 if target month is December", "target_is_academic_fee_month": "1 if target month is Jun or Jul",
        "target_next_month_expenses": "TARGET: net expenses (completed expenses - linked refunds, transfers excluded) in the following month, INR",
    }
    for cat in EXP_CATS:
        fd[f"cat_{slug(cat)}_m0"] = f"Net spend in category '{cat}' in the prediction month"
        fd[f"cat_{slug(cat)}_roll3"] = f"3-month mean net spend in category '{cat}' up to the prediction month"
    for c_ in fc_cols:
        R.append(dict(file="monthly_expense_forecasting.csv", column=c_, dtype="float" if c_ not in ("user_id", "split", "employment_type") else "string",
                      required="optional" if c_ in ("exp_lag1", "exp_lag2", "exp_lag3", "exp_target_month_last_year", "exp_roll_std_6",
                                                    "avg_txn_amount_m0", "budget_utilization_m0") else "required",
                      description=fd[c_]))
    return pd.DataFrame(R)


def methodology_text(st):
    return f"""# Dataset generation methodology

**Project:** AI Personal Finance and Smart Expense Predictor (Team 3, WeIntern minor project)
**Everything here is synthetic.** No real person, account, card, merchant or transaction is represented; merchant names are fictional.
Seed {SEED}; run `python generate_dataset.py` to reproduce byte-identical CSVs (same library versions).

## 1. Scope
150 users, 48 months (2022-01-01 to 2025-12-31), INR. Rows in transactions.csv: **{st['n_tx']:,}** ({st['n_failed']:,} failed attempts, {st['n_ref']:,} refunds).

## 2. Generation pipeline
1. **Users**: employment mix (80 salaried, 20 freelancer, 18 business owner, 10 gig, 10 student, 12 retired), age by employment, city/state (tier-weighted), household size, income (log-normal around occupation medians), one of four behaviour archetypes (disciplined_saver, balanced, overspender, inconsistent) that sets the spend-to-income ratio, saving fraction, noise level, stability and budget bias.
2. **Recurring patterns first** (salary/pension/allowance, rent, maintenance, electricity, water, gas, internet, mobile, subscriptions, insurance, loan EMIs, tuition, SIP/RD/PPF). Each pattern is a schedule with a start month, frequency, due day, and annual amount steps (e.g. April rent/salary revisions). Transactions for these are produced **only from the schedule**, so recurring_transactions.csv and transactions.csv cannot disagree or duplicate.
3. **Variable spending** per subcategory: monthly expected amount = user pool x taste weight x inflation (bounded annual rates, e.g. food 5.5%, electronics -2%) x income-lifestyle elasticity x seasonal factor x AR(1) shock (stronger for discretionary than essential spend). The count of entries per month follows a stability-controlled Poisson; amounts are log-normal.
4. **Seasonality (category-specific only)**: festival shopping (clothing, electronics, gifts, household; state-specific regional festival plus Diwali), travel peaks (Apr-Jun, Dec), monsoon health/taxi, academic fees (Jun-Jul, Jan), electricity by climate zone, business-income seasonality for seasonal owners.
5. **Shocks and drift**: 2.5-5% monthly chance of an unexpected large expense, 15% of users have a lifestyle shift between months 12-36, 18% of salaried users change employer with a raise, bonus/incentive income for some.
6. **Failed payments and refunds**: ~0.7% of digital expenses get a preceding `failed` attempt (excluded from totals); selected shopping/travel expenses (not cash) get a refund (35% partial) linked via `related_transaction_id`.
7. **Budgets**: per user one style (fixed_monthly 45%, seasonal_adjusted 35%, income_percentage 20%) x archetype bias (overspenders under-budget, savers pad); only some categories are budgeted; amounts are rounded and never identical across users.
8. **Assembly**: exact-duplicate detection (removed: {st['dup_removed']}), UPI > INR 1 lakh re-labelled as bank transfer ({st['upi_relabelled']} rows), ids assigned after sorting by date/time, derived calendar fields computed from the date.

## 3. Taxonomy
Expense categories (12): Housing (rent, maintenance, home_repairs), Food (groceries, restaurants, food_delivery), Transportation (public_transport, fuel, taxi, vehicle_maintenance, vehicle_insurance), Utilities (electricity, water, gas, internet, mobile_recharge), Healthcare (consultation, medicine, diagnostic_tests, insurance), Education (tuition, books, online_courses), Shopping (clothing, electronics, household_goods), Entertainment (movies, subscriptions, events), Personal Care (grooming, toiletries), Travel (train_bus_ticket, flights, hotel_stay), Gifts & Donations (gifts, donations), Financial Expenses (bank_fees, loan_payment, interest_charges).
Income: salary, freelance_work, business_income, incentives, other_income. Savings and Investments (transfers): sip_mutual_fund, recurring_deposit, ppf_contribution, fixed_deposit, savings_withdrawal. Transfers: own_account_transfer, atm_cash_withdrawal.

## 4. Accounting definitions (used in every derived table)
- Only `payment_status = completed` rows count. Failed rows are ignored everywhere.
- **Transfers are neither income nor expense.** Own-account moves, ATM withdrawals (cash expenses are the spend) and savings contributions/redemptions are excluded from income and expenses. Savings contributions are reported in `savings_amount`.
- **Refunds** reduce eligible spend: `net_expenses = total_expenses - refunds_received` (refund carries the original's category, so category net spend is also net of refunds). Refunds are never income.
- `net_cash_flow = total_income - net_expenses`.
- `savings_amount = SIP/RD/PPF/FD contributions - redemptions`; `savings_rate = savings_amount / total_income` (blank when income = 0).
- `essential_expenses` = expenses flagged is_essential (needs); `discretionary_expenses = total_expenses - essential_expenses` (gross, before refunds).
- Budget metrics compare **net category spend** with the category budget; `budget_utilization_percentage = 100 x net spend in budgeted categories / total_budget`.
- `month_end_balance = opening_balance_estimate + cumulative(total_income - net_expenses - savings_amount)` is a liquid-balance tracker, **not net worth** (no assets, liabilities or card dues are modelled).
- Financial year follows the Indian April-March convention; `quarter` is the calendar quarter.

## 5. Missing values
Blank = intentional and documented: unknown time on manual cash entries, unknown merchant on informal cash payments, non-applicable recurring/refund link fields, active recurring patterns (no end date), undefined ratios (zero income / no budget / no expenses), and lag features without enough history. Nothing is auto-imputed in the files; the validation report lists every count.

## 6. Forecasting table (`monthly_expense_forecasting.csv`)
- One row per user per prediction month t (Jan 2022 - Nov 2025); **target = net expenses in month t+1**.
- Features use only data through the end of month t: current/lagged/rolling net expenses, category-level net spend, income, savings, transaction counts, budgets (the target month's budget is set before that month begins), expected recurring bills due next month (schedules are known in advance), and calendar-only seasonal indicators of the target month.
- No future expenses, future counts or target-derived values are used as features. `target_*` columns are calendar facts only.
- **Split by target month, never random**: train = 2022-2023, validation = 2024, test = 2025. Early-test rows legitimately use 2024 history.
- Baselines: last month, 12-month historical average, seasonal naive (same month last year). Models: Linear Regression and Random Forest (median imputation; scaling for linear). Metrics: MAE, RMSE, MAPE (targets >= INR 1,000). Results are in dataset_validation_report.txt section 6. Test results come from models refit on train+validation; the test period is never used for fitting.
- Category-level forecasting: category net spend columns for every user-month are included; build per-category targets by shifting those columns by -1 within user.

## 7. Documented unusual-spending screen (rule-based, not a fraud label)
For a user-month with at least 12 months of history, flag `unusually_high` when `net_expenses > rolling_mean_12 + 3 x rolling_std_12` (computed on prior 12 months only). No ground-truth anomaly or fraud label is provided because no defensible labelling process exists for synthetic data; injected shocks (`planned_or_unplanned = unplanned`, descriptions starting 'Unexpected:') can be used to sanity-check recall but are not guaranteed anomalies.

## 8. Analysis hints
Income vs expenses: monthly_financial_summary. Category trends: transactions grouped by category/month. Essential vs discretionary: is_essential or needs_or_wants. Repeated overruns: categories_over_budget by user. Recurring vs non-recurring: is_recurring. Income variability: std/mean of total_income per user. Festival spending: target_festival_index / transaction dates around the festival windows in the script (`FESTIVALS`). Average-ticket changes: average_transaction_amount over time.

## 9. Limitations
Synthetic rules, not real behaviour; sparse logging of small purchases because volume was calibrated to 30-50k rows; simplified festival calendar; no net-worth modelling; model accuracy here does not transfer to real data. See validation report section 8.
"""


def readme_text(st, ml_lines):
    return f"""# AI Personal Finance and Smart Expense Predictor - synthetic dataset (Team 3, WeIntern)

Synthetic, reproducible (seed {SEED}) personal-finance data: 150 anonymous users, 2022-01-01 to 2025-12-31, INR. **No real data.**

| File | Contents |
|---|---|
| transactions.csv | {st['n_tx']:,} rows; income / expense / refund / transfer |
| users.csv | 150 profiles |
| monthly_budgets.csv | {st['n_bud']:,} category budgets |
| recurring_transactions.csv | {st['n_rec']:,} recurring-pattern segments (drive the recurring rows in transactions.csv) |
| monthly_financial_summary.csv | 7,200 user-months, reconciled to transactions |
| monthly_expense_forecasting.csv | {st['n_fc']:,} rows, leakage-safe features, target = next-month net expenses, `split` column |
| data_dictionary.csv | every column of every file |
| dataset_validation_report.txt | integrity, reconciliation, distributions, ML results, limitations |
| dataset_generation_methodology.md | rules, formulas, definitions |
| generate_dataset.py | full generator (`python generate_dataset.py --out DIR [--no-ml]`) |

Quick start:
```python
import pandas as pd
tx = pd.read_csv("transactions.csv", parse_dates=["transaction_date"])
done = tx[tx.payment_status == "completed"]          # always exclude failed rows
fc = pd.read_csv("monthly_expense_forecasting.csv")
train, val, test = [fc[fc.split == s] for s in ("train", "validation", "test")]
```
Key rules: only `completed` rows count; transfers are never income/expense; refunds reduce spend (`net_expenses`); `month_end_balance` is not net worth.
Forecast experiment (time-aware split; see report): baselines vs Linear Regression vs Random Forest. Synthetic-data accuracy is not real-world accuracy.

Forecast results snapshot:
```
{chr(10).join(ml_lines[:0] or [])}{chr(10).join(l for l in ml_lines if l.strip())}
```
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset_output")
    ap.add_argument("--no-ml", action="store_true")
    args = ap.parse_args()
    out = args.out
    os.makedirs(out, exist_ok=True)
    t0 = _time.time()
    grng = np.random.default_rng(SEED)
    users = build_users(grng)
    keygen = itertools.count(1)
    all_rows, all_rec, all_bud, urows = [], [], [], []
    for u in users:
        rng = np.random.default_rng([SEED, u["idx"]])
        r, rc, b, inc_exp = gen_user(u, rng, keygen)
        all_rows += r
        all_rec += rc
        all_bud += b
        urows.append(dict(user_id=u["user_id"], age=u["age"], occupation=u["occupation"], employment_type=u["employment_type"],
                          household_size=u["hh"], city=u["city"], state=u["state"],
                          monthly_income_estimate=float(round(inc_exp.mean() / 500) * 500),
                          income_frequency=u["income_frequency"], preferred_payment_method=u["pref_method"],
                          financial_profile=u["profile"], profile_created_date=u["profile_created"].strftime("%Y-%m-%d"),
                          opening_balance_estimate=float(round(u["opening_mult"] * inc_exp.mean() / 100) * 100)))
    print("generated rows: %d (%.1fs)" % (len(all_rows), _time.time() - t0))
    tx, prep = assemble_transactions(all_rows)
    users_df = pd.DataFrame(urows)
    bud = pd.DataFrame(all_bud).sort_values(["user_id", "year", "month", "category"]).reset_index(drop=True)
    bud.insert(0, "budget_id", ["BUD%06d" % (i + 1) for i in range(len(bud))])
    bud["budget_created_date"] = pd.to_datetime(bud["budget_created_date"]).dt.strftime("%Y-%m-%d")
    rec = pd.DataFrame(all_rec)
    rec["start_date"] = pd.to_datetime(rec["start_date"]).dt.strftime("%Y-%m-%d")
    rec["end_date"] = pd.to_datetime(rec["end_date"]).dt.strftime("%Y-%m-%d")
    rec = rec[["recurring_id", "user_id", "transaction_type", "category", "subcategory", "description", "expected_amount", "frequency",
               "start_date", "end_date", "expected_day_of_month", "active_status"]].sort_values("recurring_id").reset_index(drop=True)
    tx.to_csv(f"{out}/transactions.csv", index=False, encoding="utf-8")
    users_df.to_csv(f"{out}/users.csv", index=False, encoding="utf-8")
    bud.to_csv(f"{out}/monthly_budgets.csv", index=False, encoding="utf-8")
    rec.to_csv(f"{out}/recurring_transactions.csv", index=False, encoding="utf-8")
    # re-read saved files so everything downstream is derived from what was actually written
    txr = pd.read_csv(f"{out}/transactions.csv", parse_dates=["transaction_date"], low_memory=False)
    usr = pd.read_csv(f"{out}/users.csv")
    budr = pd.read_csv(f"{out}/monthly_budgets.csv")
    recr = pd.read_csv(f"{out}/recurring_transactions.csv", parse_dates=["start_date", "end_date"])
    S, cn = build_summary(txr, usr, budr)
    S.to_csv(f"{out}/monthly_financial_summary.csv", index=False, encoding="utf-8")
    fc = build_forecast(S, cn, usr, recr)
    fc.to_csv(f"{out}/monthly_expense_forecasting.csv", index=False, encoding="utf-8")
    print("tables written (%.1fs)" % (_time.time() - t0))
    ml_lines = []
    if not args.no_ml:
        ml_lines, _ = run_ml(fc)
        print("\n".join(ml_lines))
    txr["transaction_date"] = txr["transaction_date"].dt.strftime("%Y-%m-%d")
    txv = pd.read_csv(f"{out}/transactions.csv", low_memory=False)
    txv["transaction_date"] = pd.to_datetime(txv["transaction_date"])
    Sr = pd.read_csv(f"{out}/monthly_financial_summary.csv")
    fcr = pd.read_csv(f"{out}/monthly_expense_forecasting.csv")
    recv = pd.read_csv(f"{out}/recurring_transactions.csv")
    txv2 = txv.copy()
    txv2["transaction_date"] = txv2["transaction_date"].dt.strftime("%Y-%m-%d")
    report, fails = validate(out, txv2, usr, budr, recv, Sr, fcr, prep, ml_lines)
    with open(f"{out}/dataset_validation_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    dd = data_dictionary(list(fcr.columns))
    dd.to_csv(f"{out}/data_dictionary.csv", index=False, encoding="utf-8")
    # dictionary coverage check
    miss = []
    for fn, df_ in (("transactions.csv", txv2), ("users.csv", usr), ("monthly_budgets.csv", budr), ("recurring_transactions.csv", recv),
                    ("monthly_financial_summary.csv", Sr), ("monthly_expense_forecasting.csv", fcr)):
        known = set(dd[dd.file == fn].column)
        miss += [f"{fn}:{c_}" for c_ in df_.columns if c_ not in known]
    st = dict(n_tx=len(txv2), n_failed=int((txv2.payment_status == "failed").sum()), n_ref=int((txv2.transaction_type == "refund").sum()),
              n_bud=len(budr), n_rec=len(recv), n_fc=len(fcr), **prep)
    with open(f"{out}/dataset_generation_methodology.md", "w", encoding="utf-8") as f:
        f.write(methodology_text(st))
    with open(f"{out}/README.md", "w", encoding="utf-8") as f:
        f.write(readme_text(st, ml_lines))
    me = os.path.abspath(__file__)
    if os.path.abspath(os.path.join(out, "generate_dataset.py")) != me:
        shutil.copy(me, os.path.join(out, "generate_dataset.py"))
    names = ["transactions.csv", "users.csv", "monthly_budgets.csv", "recurring_transactions.csv", "monthly_financial_summary.csv",
             "monthly_expense_forecasting.csv", "data_dictionary.csv", "dataset_validation_report.txt",
             "dataset_generation_methodology.md", "README.md", "generate_dataset.py"]
    with zipfile.ZipFile(f"{out}/ai_personal_finance_dataset.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.write(os.path.join(out, n), n)
    print("\n".join(report[-40:]))
    print("dictionary gaps:", miss)
    print("fails:", fails, "| total %.1fs" % (_time.time() - t0))
    if fails or miss:
        sys.exit(1)


if __name__ == "__main__":
    main()
