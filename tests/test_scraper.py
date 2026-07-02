from app.scraper import _is_job_solution_bundle, rows_to_assessments
from app.utils import parse_duration_minutes
from app.utils import test_type_letter_from_label as type_letter_from_label


def test_parse_duration_minutes_handles_all_seed_formats():
    assert parse_duration_minutes("Approximate Completion Time in minutes = 30") == 30
    assert parse_duration_minutes("13.0") == 13
    assert parse_duration_minutes("Approximate Completion Time in minutes = Untimed") is None
    assert parse_duration_minutes("") is None


def test_test_type_letter_mapping():
    assert type_letter_from_label("Knowledge & Skills") == "K"
    assert type_letter_from_label("Personality & Behavior") == "P"
    assert type_letter_from_label("unknown label") == ""


def test_job_solution_bundle_detection():
    assert _is_job_solution_bundle("Entry Level Sales Solution")
    assert not _is_job_solution_bundle("Core Java (Advanced Level) (New)")


def test_rows_to_assessments_excludes_job_solutions_and_fills_fields():
    rows = [
        {
            "data-entity-id": "1",
            "Assessment Name": "Core Java (New)",
            "Relative URL": "https://www.shl.com/solutions/products/product-catalog/view/core-java-new/",
            "Remote Testing": "Yes",
            "Adaptive/IRT": "No",
            "Test Type": "Knowledge & Skills",
            "Assessment Length": "Approximate Completion Time in minutes = 15",
        },
        {
            "data-entity-id": "2",
            "Assessment Name": "Entry Level Sales Solution",
            "Relative URL": "https://www.shl.com/solutions/products/product-catalog/view/entry-level-sales-solution/",
            "Remote Testing": "Yes",
            "Adaptive/IRT": "No",
            "Test Type": "",
            "Assessment Length": "",
        },
    ]
    assessments = rows_to_assessments(rows)
    assert len(assessments) == 1
    a = assessments[0]
    assert a.name == "Core Java (New)"
    assert a.test_type == "K"
    assert a.duration_minutes == 15
    assert a.remote_testing is True
    assert "Java" in a.skills
