from opensearchpy import Document, Q
from opensearchpy.helpers import query, search
import pytest


#@pytest.mark.skip
def test_neural_query_to_dict_simple():
    s = search.Search(index="test index")

    s = s.query(
        Q(
            "neural",
            embedding_field="passage_embedding",
            query_text="wild west",
            model_id="aVeif4oB5Vm0Tdw8zYO2",
            k=5,
        )
    )

    expected_output = {
        "query": {
            "neural": {
                "passage_embedding": {
                    "query_text": "wild west",
                    "model_id": "aVeif4oB5Vm0Tdw8zYO2",
                    "k": 5,
                }
            }
        }
    }

    assert s.to_dict() == expected_output


#@pytest.mark.skip
def test_neural_complex_example() -> None:
    s = search.Search()
    s = (
        s.query("match", title="python")
        .query(
            Q(
                "neural",
                embedding_field="passage_embedding",
                query_text="wild west",
                model_id="aVeif4oB5Vm0Tdw8zYO2",
                k=5,
            )
        )
        .query(~Q("match", title="ruby"))
        .filter(Q("term", category="meetup") | Q("term", category="conference"))
        .post_filter("terms", tags=["prague", "czech"])
        .script_fields(more_attendees="doc['attendees'].value + 42")
    )

    s.aggs.bucket("per_country", "terms", field="country").metric(
        "avg_attendees", "avg", field="attendees"
    )

    s.query.minimum_should_match = 2

    s = s.highlight_options(order="score").highlight("title", "body", fragment_size=50)

    assert {
        "query": {
            "bool": {
                "filter": [
                    {
                        "bool": {
                            "should": [
                                {"term": {"category": "meetup"}},
                                {"term": {"category": "conference"}},
                            ]
                        }
                    }
                ],
                "must": [
                    {"match": {"title": "python"}},
                    {
                        "neural": {
                            "passage_embedding": {
                                "query_text": "wild west",
                                "model_id": "aVeif4oB5Vm0Tdw8zYO2",
                                "k": 5,
                            }
                        }
                    },
                ],
                "must_not": [{"match": {"title": "ruby"}}],
                "minimum_should_match": 2,
            }
        },
        "post_filter": {"terms": {"tags": ["prague", "czech"]}},
        "aggs": {
            "per_country": {
                "terms": {"field": "country"},
                "aggs": {"avg_attendees": {"avg": {"field": "attendees"}}},
            }
        },
        "highlight": {
            "order": "score",
            "fields": {"title": {"fragment_size": 50}, "body": {"fragment_size": 50}},
        },
        "script_fields": {"more_attendees": {"script": "doc['attendees'].value + 42"}},
    } == s.to_dict()


#@pytest.mark.skip
def test_neural():
    q = Q(
        "neural",
        embedding_field="passage_embedding",
        query_text="wild west",
        model_id="aVeif4oB5Vm0Tdw8zYO2",
        k=5,
    )
    print('%%%')
    print(q.to_dict())

# Test case for error handling when no parameters are provided
def test_neural_query_empty_params():
    with pytest.raises(ValueError, match="No valid parameters provided"):
        Q("neural")

# Test case for error handling when 'embedding_field' is missing
def test_neural_query_missing_embedding_field():
    with pytest.raises(ValueError, match="Missing 'embedding_field' parameter"):
        Q(
            "neural",
            query_text="wild west",
            model_id="aVeif4oB5Vm0Tdw8zYO2",
            k=5
        )

# Test case for a neural query with additional parameters
def test_neural_query_additional_params():
    q = Q(
        "neural",
        embedding_field="passage_embedding",
        query_text="wild west",
        model_id="aVeif4oB5Vm0Tdw8zYO2",
        k=10,
        threshold=0.7
    )
    expected_output = {
        "neural": {
            "passage_embedding": {
                "query_text": "wild west",
                "model_id": "aVeif4oB5Vm0Tdw8zYO2",
                "k": 10,
                "threshold": 0.7
            }
        }
    }
    assert q.to_dict() == expected_output

# Test case for neural query within a combined query
def test_neural_combined_query():
    s = search.Search(index="test index")
    s = s.query(
        Q("bool",
          must=[
              Q("match", title="AI"),
              Q(
                  "neural",
                  embedding_field="document_embedding",
                  query_text="deep learning",
                  model_id="xyz123",
                  k=3
              )
          ],
          must_not=[Q("match", title="hardware")]
        )
    )
    expected_output = {
        "query": {
            "bool": {
                "must": [
                    {"match": {"title": "AI"}},
                    {
                        "neural": {
                            "document_embedding": {
                                "query_text": "deep learning",
                                "model_id": "xyz123",
                                "k": 3
                            }
                        }
                    }
                ],
                "must_not": [{"match": {"title": "hardware"}}]
            }
        }
    }
    assert s.to_dict() == expected_output

# Test case for neural query within an aggregation
def test_neural_aggregation_example():
    s = search.Search(index="test index")
    s = s.query(
        Q(
            "neural",
            embedding_field="content_embedding",
            query_text="data science",
            model_id="abc789",
            k=4
        )
    ).aggs.bucket("category_count", "terms", field="category")
    expected_output = {
        "query": {
            "neural": {
                "content_embedding": {
                    "query_text": "data science",
                    "model_id": "abc789",
                    "k": 4
                }
            }
        },
        "aggs": {
            "category_count": {
                "terms": {"field": "category"}
            }
        }
    }
    assert s.to_dict() == expected_output

# Test case for a neural query with a missing required key inside a nested dictionary
def test_neural_missing_key_in_params():
    with pytest.raises(ValueError, match="Missing 'query_text' key"):
        q = Q(
            "neural",
            embedding_field="passage_embedding",
            model_id="aVeif4oB5Vm0Tdw8zYO2",
            k=5
        )
        q.to_dict()

