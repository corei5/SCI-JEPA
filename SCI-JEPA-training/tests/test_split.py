"""Unit tests of the stored split's balancing and schema logic (no data needed)."""
import pyarrow as pa
import pyarrow.parquet as pq

from sci_jepa.data.split import _as_text, balanced_quota, domain_weights, unified_schema


def test_quota_follows_shares_and_redistributes():
    w = {"A": 4.0, "B": 1.0, "C": 1.0}
    q = balanced_quota(w, {"A": 1000, "B": 1000, "C": 2}, 60)
    assert sum(q.values()) == 60
    assert q["C"] == 2                     # a small domain gives what it cannot fill to the others
    assert q["A"] > q["B"] > q["C"]


def test_quota_zero_means_everything():
    assert balanced_quota({"A": 1.0}, {"A": 7, "B": 3}, 0) == {"A": 7, "B": 3}


def test_same_shares_in_every_split():
    pooled = {"Physics": 900, "CS": 600, "Chem": 20}
    w, alpha = domain_weights(pooled, "temperature", 0.5)
    assert alpha == 0.5
    train = balanced_quota(w, {"Physics": 800, "CS": 300, "Chem": 15}, 100)
    test = balanced_quota(w, {"Physics": 100, "CS": 300, "Chem": 15}, 100)   # every domain can fill its share
    assert abs(train["Physics"] - test["Physics"]) <= 1 and abs(train["CS"] - test["CS"]) <= 1


def test_uniform_and_natural_weights():
    pooled = {"a": 100, "b": 1}
    assert domain_weights(pooled, "uniform", 0.5)[0] == {"a": 1.0, "b": 1.0}
    assert domain_weights(pooled, "natural", 0.5)[0] == {"a": 100.0, "b": 1.0}


def test_mixed_column_types_become_text(tmp_path):
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    pq.write_table(pa.table({"oa_year": pa.array(["2017"]), "x": pa.array([1])}), a)
    pq.write_table(pa.table({"oa_year": pa.array([2018.0]), "x": pa.array([2])}), b)
    schema, to_text = unified_schema([str(a), str(b)])
    assert to_text == {"oa_year"} and pa.types.is_large_string(schema.field("oa_year").type)
    assert _as_text(2018.0) == "2018" and _as_text(True) == "True" and _as_text(None) is None
