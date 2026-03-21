"""
Complex Dependent Test Datasets:
  - customers.csv   (500 rows) — customer demographics + churn label
  - orders.csv      (2000 rows) — order transactions referencing customer_id
  - products.csv    (100 rows) — product catalog referenced by orders

Relationships: orders.customer_id → customers.customer_id
               orders.product_id  → products.product_id

Intentional data issues:
  - Missing values in multiple columns
  - Outlier revenue values
  - Duplicate order IDs
  - Inconsistent date formats
  - Skewed distributions
"""
import random, math, csv
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)
OUT = Path(__file__).parent / "fixtures"
OUT.mkdir(exist_ok=True)

REGIONS   = ["North", "South", "East", "West", "Central"]
SEGMENTS  = ["Consumer", "Corporate", "SMB", "Enterprise"]
PAYMENT   = ["Credit Card", "PayPal", "Bank Transfer", "Cash", None]
CATEGORIES = ["Electronics", "Apparel", "Home", "Sports", "Books", "Toys"]
PRODUCTS   = [
    ("P{:04d}".format(i), random.choice(CATEGORIES),
     round(random.uniform(5, 2000), 2),
     round(random.uniform(0.05, 0.55), 2))
    for i in range(1, 101)
]

START_DATE = datetime(2022, 1, 1)


def rnd_date(fmt="%Y-%m-%d", noise_days=730):
    d = START_DATE + timedelta(days=random.randint(0, noise_days))
    if fmt == "mixed":
        fmt = random.choice(["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y"])
    return d.strftime(fmt)


print("Generating customers.csv …")
with open(OUT / "customers.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["customer_id","age","gender","region","segment",
                "tenure_months","annual_spend","support_calls",
                "satisfaction_score","churn"])
    for i in range(1, 501):
        age     = random.randint(18, 75) if random.random() > 0.03 else None
        gender  = random.choice(["M","F","Other",None])
        region  = random.choice(REGIONS)
        seg     = random.choice(SEGMENTS)
        tenure  = random.randint(1, 84)
        # bimodal spend: loyal high-spenders vs at-risk low-spenders
        if random.random() > 0.35:
            spend = round(random.gauss(12000, 4000), 2)
        else:
            spend = round(random.gauss(2500, 800), 2)
        spend = max(100, spend) if spend else None
        if random.random() < 0.04: spend = None
        calls  = random.randint(0, 15) if random.random() > 0.05 else None
        score  = round(random.uniform(1, 5), 1) if random.random() > 0.08 else None
        # churn correlated with low score + high calls
        base_churn = 0.1
        if score and score < 2.5: base_churn += 0.3
        if calls  and calls > 8:  base_churn += 0.2
        if tenure < 6:            base_churn += 0.15
        churn = 1 if random.random() < base_churn else 0
        w.writerow([f"C{i:05d}", age, gender, region, seg,
                    tenure, spend, calls, score, churn])

print("Generating products.csv …")
with open(OUT / "products.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["product_id","product_name","category",
                "unit_price","cost_price","margin","weight_kg",
                "in_stock","rating"])
    for pid, cat, price, margin in PRODUCTS:
        name    = f"{cat} Item {pid}"
        cost    = round(price * (1 - margin), 2)
        weight  = round(random.uniform(0.1, 15), 2) if random.random() > 0.06 else None
        in_stock= random.choice([1,1,1,0])
        rating  = round(random.uniform(1,5),1) if random.random() > 0.1 else None
        w.writerow([pid, name, cat, price, cost,
                    round(margin,3), weight, in_stock, rating])

print("Generating orders.csv …")
with open(OUT / "orders.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["order_id","customer_id","product_id","order_date",
                "ship_date","quantity","unit_price","discount",
                "revenue","payment_method","returned"])
    order_ids = []
    for i in range(1, 2001):
        oid   = f"ORD{i:06d}"
        # 3% duplicate order IDs (data quality issue)
        if random.random() < 0.03 and order_ids:
            oid = random.choice(order_ids)
        else:
            order_ids.append(oid)
        cid   = f"C{random.randint(1,500):05d}"
        pid, cat, base_price, mg = random.choice(PRODUCTS)
        odate = rnd_date("mixed")
        # ship date sometimes missing (3%)
        sdate = rnd_date() if random.random() > 0.03 else None
        qty   = random.randint(1, 20)
        # occasional outlier price
        price = base_price if random.random() > 0.02 else base_price * random.uniform(8, 25)
        disc_raw = random.choice([0,0,0,0.05,0.10,0.15,0.20,0.25,None])
        disc  = (round(disc_raw, 2) if disc_raw is not None else None) if random.random() > 0.04 else None
        disc_val = disc or 0
        rev   = round(qty * price * (1 - disc_val), 2)
        # 5% negative revenue (returns entered wrong)
        if random.random() < 0.05: rev = -rev
        pay   = random.choice(PAYMENT)
        ret   = 1 if random.random() < 0.08 else 0
        w.writerow([oid, cid, pid, odate, sdate, qty,
                    round(price,2), disc, rev, pay, ret])

print("\n✅  Generated:")
for p in ["customers.csv","products.csv","orders.csv"]:
    size = (OUT/p).stat().st_size
    print(f"   tests/fixtures/{p}  {size//1024} KB")
