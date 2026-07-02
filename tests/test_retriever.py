def test_catalog_loads_and_excludes_job_solutions(shared_retriever):
    all_items = shared_retriever.get_all()
    assert len(all_items) > 100
    assert all(not a.name.strip().lower().endswith("solution") for a in all_items)


def test_search_returns_ranked_relevant_results(shared_retriever):
    results = shared_retriever.search("Java software developer", k=10)
    assert results
    names = [a.name.lower() for a, _ in results]
    assert any("java" in n for n in names)


def test_get_by_name_resolves_acronyms(shared_retriever):
    opq = shared_retriever.get_by_name("OPQ")
    gsa = shared_retriever.get_by_name("GSA")
    assert opq is not None and "OPQ" in opq.name
    assert gsa is not None and "Global Skills" in gsa.name


def test_get_by_name_returns_none_for_unknown(shared_retriever):
    assert shared_retriever.get_by_name("TotallyMadeUpAssessmentXYZ") is None


def test_apply_filters_duration_and_remote(shared_retriever):
    items = shared_retriever.get_all()
    filtered = shared_retriever.apply_filters(items, max_duration_minutes=10, remote_required=True)
    assert all(a.remote_testing for a in filtered)
    assert all(a.duration_minutes is None or a.duration_minutes <= 10 for a in filtered)


def test_every_url_is_unique(shared_retriever):
    urls = [a.url for a in shared_retriever.get_all()]
    assert len(urls) == len(set(urls))
