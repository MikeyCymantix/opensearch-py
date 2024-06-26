from opensearchpy import Q
from opensearchpy.helpers import search
import pytest


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


def test_neural_query_empty_params():
    with pytest.raises(
        ValueError,
        match="Expected a single embedding field key, but got multiple or none.",
    ):
        Q("neural")


def test_neural_query_multimodal_image():
    q = Q(
        "neural",
        embedding_field="passage_embedding",
        query_text="wild west",
        query_image="123481234.jpg",
        model_id="aVeif4oB5Vm0Tdw8zYO2",
        k=10,
    )
    expected_output = {
        "neural": {
            "passage_embedding": {
                "query_text": "wild west",
                "model_id": "aVeif4oB5Vm0Tdw8zYO2",
                "k": 10,
                "query_image": "123481234.jpg",
            }
        }
    }
    assert q.to_dict() == expected_output


def test_neural_combined_query():
    s = search.Search(index="test index")
    s = s.query(
        Q(
            "bool",
            must=[
                Q("match", title="AI"),
                Q(
                    "neural",
                    embedding_field="document_embedding",
                    query_text="deep learning",
                    model_id="xyz123",
                    k=3,
                ),
            ],
            must_not=[Q("match", title="hardware")],
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
                                "k": 3,
                            }
                        }
                    },
                ],
                "must_not": [{"match": {"title": "hardware"}}],
            }
        }
    }
    assert s.to_dict() == expected_output


def test_neural_missing_key_in_params():
    with pytest.raises(KeyError, match="Missing query_text key"):
        q = Q(
            "neural",
            embedding_field="passage_embedding",
            model_id="aVeif4oB5Vm0Tdw8zYO2",
            k=5,
        )
        q.to_dict()
