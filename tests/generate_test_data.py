"""
Generate synthetic test datasets for Maya AI Agent testing.
Creates:
  - sample_sales.csv:     200 rows of sales data with intentional quality issues
  - sample_customers.csv: 50 rows for multi-file merge testing
"""
import os
import random
import csv
from datetime import datetime, timedelta

random.seed(42)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
os.makedirs(FIXTURES_DIR, exist_ok=True)

# ── Sales Dataset ─────────────────────────────────────────────────────
products = [
    ("P001", "Laptop Pro", "Electronics", 999.99),
    ("P002", "Wireless Mouse", "Electronics", 29.99),
    ("P003", "Office Chair", "Furniture", 349.00),
    ("P004", "Standing Desk", "Furniture", 599.00),
    ("P005", "Notebook Pack", "Stationery", 12.50),
    ("P006", "Monitor 27\"", "Electronics", 449.99),
    ("P007", "Keyboard Mech", "Electronics", 89.99),
    ("P008", "Desk Lamp", "Furniture", 45.00),
    ("P009", "Pen Set", "Stationery", 8.99),
    ("P010", "Webcam HD", "Electronics", 79.99),
]

regions = ["North", "South", "East", "West", "Central"]
customer_ids = [f"C{str(i).zfill(3)}" for i in range(1, 51)]

start_date = datetime(2024, 1, 1)
sales_rows = []

for i in range(200):
    pid, pname, cat, price = random.choice(products)
    qty = random.randint(1, 20)
    discount = round(random.choice([0, 0, 0, 5, 10, 15, 20, 25]), 1)
    total = round(qty * price * (1 - discount / 100), 2)
    date = start_date + timedelta(days=random.randint(0, 364))
    region = random.choice(regions)
    cid = random.choice(customer_ids)

    row = {
        "date": date.strftime("%Y-%m-%d"),
        "product_id": pid,
        "product_name": pname,
        "category": cat,
        "quantity": qty,
        "unit_price": price,
        "total_revenue": total,
        "customer_id": cid,
        "region": region,
        "discount_pct": discount,
    }

    # Inject data quality issues
    if i == 5:
        row["quantity"] = ""          # missing numeric
    if i == 12:
        row["region"] = "  North "   # whitespace issues
    if i == 18:
        row["product_name"] = None    # null
    if i == 25:
        row["total_revenue"] = -50.0  # negative revenue
    if i == 30:
        row["date"] = "invalid-date"  # bad date
    if i == 42:
        row["discount_pct"] = 150     # out of range
    if i in [50, 51]:
        # exact duplicate rows
        row = sales_rows[50] if i == 51 and len(sales_rows) > 50 else row

    sales_rows.append(row)

sales_path = os.path.join(FIXTURES_DIR, "sample_sales.csv")
with open(sales_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(sales_rows[0].keys()))
    writer.writeheader()
    for row in sales_rows:
        writer.writerow(row)

print(f"✅ Created {sales_path} ({len(sales_rows)} rows)")

# ── Customers Dataset ─────────────────────────────────────────────────
first_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace",
               "Henry", "Iris", "Jack", "Karen", "Leo", "Mona", "Nick", "Olivia"]
last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
              "Miller", "Davis", "Rodriguez", "Martinez"]
cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
          "San Antonio", "Dallas", "San Jose", "Austin", "Denver"]

customer_rows = []
for i in range(50):
    cid = f"C{str(i + 1).zfill(3)}"
    name = f"{random.choice(first_names)} {random.choice(last_names)}"
    age = random.randint(22, 65)
    city = random.choice(cities)
    signup = start_date - timedelta(days=random.randint(30, 730))
    tier = random.choice(["Bronze", "Silver", "Gold", "Platinum"])

    row = {
        "customer_id": cid,
        "customer_name": name,
        "age": age,
        "city": city,
        "signup_date": signup.strftime("%Y-%m-%d"),
        "tier": tier,
    }

    # A few quality issues
    if i == 3:
        row["age"] = -5       # negative age
    if i == 7:
        row["city"] = "  los angeles"  # case + whitespace
    if i == 15:
        row["customer_name"] = ""      # empty string

    customer_rows.append(row)

customers_path = os.path.join(FIXTURES_DIR, "sample_customers.csv")
with open(customers_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(customer_rows[0].keys()))
    writer.writeheader()
    for row in customer_rows:
        writer.writerow(row)

print(f"✅ Created {customers_path} ({len(customer_rows)} rows)")
