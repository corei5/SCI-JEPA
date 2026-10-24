"""Data: from normalized parquet to model inputs.

    ku_schema    KU types, relation types, field/domain mapping, masking templates, counterfactual edits
    ku_graph     reads the parquet, cleans KUs, balances domains, embeds texts, builds the KU graph (cached)
    graph_index  numpy views of the KU graph (GraphIndex) and the cached loader get_index(cfg)
    instances    Sampler: builds masked training / evaluation instances (leak-safe masking)
    synthetic    generated records in the parquet schema, for smoke tests
"""
