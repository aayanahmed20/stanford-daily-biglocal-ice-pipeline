"""
Integration smoke test: run extract_record() + validate_dataframe() over a
small, real sample of stanforddams/biglocal rows (title/url/topics/date +
full_text), without needing network access to Hugging Face.

The `full_text` values here are copied from the dataset's public preview,
which truncates each body to ~300 characters ("..."). That understates real
completeness/word-count numbers - this test exists to confirm the pipeline
runs cleanly end-to-end on real rows, not to certify accuracy stats. Run
`scripts/pipeline.py` against the full dataset for real numbers.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from extract import extract_record  # noqa: E402
from validate import validate_dataframe, usable_mask  # noqa: E402

SAMPLE_ROWS = [
    dict(
        url="https://www.ice.gov/news/releases/12-arrested-south-texas-worksite-enforcement-operation-ice-rio-grande-valley-federal",
        title="12 arrested in a South Texas worksite enforcement operation by ICE Rio Grande Valley, federal partners",
        topics="Worksite Enforcement",
        date_normalized="06/10/2025",
        city="Harlingen", state="TX",
        location_full_text="HARLINGEN, Texas",
        full_text=(
            "HARLINGEN, Texas \u2014 U.S. Immigration and Customs Enforcement "
            "arrested 12 illegal aliens on June 9 during a targeted worksite "
            "enforcement operation at two business locations, one in Harlingen "
            "and one in San Benito, Texas. The operation was conducted by ICE's "
            "Homeland Security Investigations Rio Grande Valley office, with "
            "federal partners."
        ),
    ),
    dict(
        url="https://www.ice.gov/news/releases/2-mexican-nationals-defendants-ice-cases-secured-arizona",
        title="2 Mexican nationals, defendants in ICE cases secured in Arizona",
        topics="Narcotics",
        date_normalized="03/06/2025",
        city="Phoenix", state="AZ",
        location_full_text="PHOENIX, Ariz.",
        full_text=(
            "PHOENIX, Ariz. \u2013 Two Mexican nationals who are targets of an "
            "ongoing U.S. Immigration and Customs Enforcement investigation "
            "appeared for their initial appearances Feb. 28, after they were "
            "secured from Mexico the previous day. Jose Bibiano Cabrera-Cabrera, "
            "37 and Jesus Humberto Limon-Lopez, 43, were taken into U.S. custody."
        ),
    ),
    dict(
        url="https://www.ice.gov/news/releases/caribbean-arms-trafficking-ringleader-charged-conspiracy-smuggle-firearms-us",
        title="Caribbean arms trafficking ringleader charged with conspiracy to smuggle firearms from US",
        topics="Firearms, Ammunition and Explosives, Contraband",
        date_normalized="02/03/2025",
        city="Tampa", state="FL",
        location_full_text="TAMPA, Fla.",
        full_text=(
            "TAMPA, Fla. \u2014 A Trinidadian national, and international "
            "firearms trafficking organization ringleader, has been indicted "
            "with conspiracy to commit smuggling and conspiracy to traffic "
            "firearms following a Homeland Security Investigations (HSI) Tampa "
            "investigation. Shem Wayne Alexander, 35, of Port of Spain, "
            "Trinidad, faces charges."
        ),
    ),
    dict(
        url="https://www.ice.gov/news/releases/canadian-national-ice-custody-passes-away",
        title="Canadian national in ICE custody passes away",
        topics="Detainee Death Notifications",
        date_normalized="06/25/2025",
        city="Miami", state="FL",
        location_full_text="MIAMI",
        full_text=(
            "MIAMI \u2014 Johnny Noviello, a 49-year-old citizen of Canada in "
            "the custody of U.S. Immigration and Customs Enforcement, was "
            "pronounced deceased by the Miami Fire Rescue Department June 23 "
            "at 1:36 p.m. The cause of death is still under investigation."
        ),
    ),
    dict(
        url="https://www.ice.gov/news/releases/duplicate-of-above-for-testing",
        title="Duplicate URL test row",
        topics="Narcotics",
        date_normalized="not a real date",
        city=None, state=None,
        location_full_text=None,
        full_text="Too short to pass the word-count filter.",
    ),
]


def test_pipeline_smoke_end_to_end():
    records = [extract_record(dict(row)) for row in SAMPLE_ROWS]
    df = pd.DataFrame(records)

    report = validate_dataframe(df)

    # Schema always passes here: url/title are present on every row.
    assert report["schema"]["passes"] is True

    # The malformed row should get caught by sanity checks.
    assert report["sanity_issues"]["unparseable_dates"] >= 1
    assert report["sanity_issues"]["empty_or_near_empty_body"] >= 1

    # The death-in-custody row should classify correctly and extract a name+age.
    death_row = df[df["url"].str.contains("canadian-national")].iloc[0]
    assert death_row["action_type"] == "death_in_custody"
    assert any(p["name"] == "Johnny Noviello" for p in death_row["extracted_people"])

    # At least the 4 well-formed rows should be usable for analysis.
    assert usable_mask(df).sum() >= 4
