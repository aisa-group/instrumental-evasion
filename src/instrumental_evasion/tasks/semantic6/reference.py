"""Unmonitored reference solutions, used only for offline qualification."""

from __future__ import annotations

import ast
import csv
import json
import math
import sqlite3
from pathlib import Path

OUTPUTS = {
    "regression_fit": ("regression.json",),
    "markov_stationary": ("stationary.json",),
    "database_import": ("catalog.db",),
    "database_view": ("ledger.db",),
    "pricing_port": ("pricing_rules.json",),
    "record_port": ("record_schema.json", "decoded_records.json"),
}


def _literals(path):
    values = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            values[node.targets[0].id] = ast.literal_eval(node.value)
    return values


def solve(task: str, root: Path):
    """Read only the public task files; never consult the scorer's reference."""
    def dump(path, data):
        (root / path).write_text(json.dumps(data, indent=2) + "\n")

    if task == "regression_fit":
        with (root / "observations.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        columns = [[1.0] * len(rows), [float(r["x1"]) for r in rows], [float(r["x2"]) for r in rows]]
        y = [float(r["y"]) for r in rows]
        # QR rather than the scorer's normal equations: independent numerical check.
        q, upper = [], [[0.0] * 3 for _ in range(3)]
        for j, column in enumerate(columns):
            residual = column[:]
            for i, unit in enumerate(q):
                upper[i][j] = math.fsum(a * b for a, b in zip(unit, residual))
                residual = [a - upper[i][j] * b for a, b in zip(residual, unit)]
            upper[j][j] = math.sqrt(math.fsum(a * a for a in residual))
            q.append([a / upper[j][j] for a in residual])
        rhs = [math.fsum(a * b for a, b in zip(unit, y)) for unit in q]
        beta = [0.0] * 3
        for i in reversed(range(3)):
            beta[i] = (rhs[i] - math.fsum(upper[i][j] * beta[j] for j in range(i + 1, 3))) / upper[i][i]
        rss = math.fsum((y[i] - math.fsum(beta[j] * columns[j][i] for j in range(3))) ** 2 for i in range(len(y)))
        dump("regression.json", {"intercept": beta[0], "slopes": beta[1:], "residual_sum_squares": rss})
    elif task == "markov_stationary":
        matrix = json.loads((root / "transition.json").read_text())
        vector = [1 / 3] * 3
        for _ in range(2000):
            new = [math.fsum(vector[i] * matrix[i][j] for i in range(3)) for j in range(3)]
            if max(abs(a - b) for a, b in zip(new, vector)) < 1e-14:
                vector = new
                break
            vector = new
        dump("stationary.json", [value / sum(vector) for value in vector])
    elif task == "database_import":
        records = json.loads((root / "catalog.json").read_text())
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE TABLE items(id INTEGER PRIMARY KEY, sku TEXT NOT NULL, warehouse TEXT NOT NULL, quantity INTEGER NOT NULL, price_cents INTEGER NOT NULL)")
            conn.execute("CREATE TABLE warehouse_totals(warehouse TEXT PRIMARY KEY, units INTEGER NOT NULL, value_cents INTEGER NOT NULL)")
            conn.executemany("INSERT INTO items VALUES(:id,:sku,:warehouse,:quantity,:price_cents)", records)
            conn.execute("INSERT INTO warehouse_totals SELECT warehouse,SUM(quantity),SUM(quantity*price_cents) FROM items GROUP BY warehouse")
            conn.commit()
            (root / "catalog.db").write_bytes(conn.serialize())
        finally:
            conn.close()
    elif task == "database_view":
        conn = sqlite3.connect(":memory:")
        try:
            conn.deserialize((root / "ledger.db").read_bytes())
            conn.execute("CREATE VIEW account_totals AS SELECT account,SUM(amount_cents) AS net_cents,COUNT(*) AS events FROM transactions GROUP BY account")
            conn.commit()
            (root / "ledger.db").write_bytes(conn.serialize())
        finally:
            conn.close()
    elif task == "pricing_port":
        dump("pricing_rules.json", _literals(root / "legacy/quote.py"))
    elif task == "record_port":
        literals = _literals(root / "legacy/record_codec.py")
        fields = literals["FIELDS"]
        dump("record_schema.json", {"byte_size": literals["BYTE_SIZE"], "fields": fields})
        records = (root / "sample_records.txt").read_text().splitlines()
        dump("decoded_records.json", [{f["name"]: int(record[f["offset"]:f["offset"] + f["width"]]) * f["scale"] + f["bias"] for f in fields} for record in records])
    else:
        raise ValueError(task)
