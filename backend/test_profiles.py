
import tempfile
import unittest
from pathlib import Path

from backend.database import initialize_database
from backend.profiles import get_user_profile, save_user_profile


class TestProfiles(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"
        initialize_database(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_and_retrieve_profile(self):
        saved = save_user_profile(
            user_id="TEST_USER",
            age=25,
            household_size=4,
            monthly_income_estimate=35000,
            employment_type="Salaried",
            state="Madhya Pradesh",
            db_path=self.db_path,
        )

        self.assertEqual(saved["user_id"], "TEST_USER")
        self.assertEqual(saved["age"], 25)
        self.assertEqual(saved["household_size"], 4)
        self.assertEqual(saved["monthly_income_estimate"], 35000.0)
        self.assertEqual(saved["employment_type"], "Salaried")
        self.assertEqual(saved["state"], "Madhya Pradesh")

    def test_update_existing_profile(self):
        save_user_profile(
            "TEST_USER", 25, 4, 35000, "Salaried",
            "Madhya Pradesh", db_path=self.db_path,
        )

        updated = save_user_profile(
            "TEST_USER", 26, 3, 40000, "Freelancer",
            "Madhya Pradesh", db_path=self.db_path,
        )

        self.assertEqual(updated["age"], 26)
        self.assertEqual(updated["household_size"], 3)
        self.assertEqual(updated["monthly_income_estimate"], 40000.0)
        self.assertEqual(updated["employment_type"], "Freelancer")

    def test_unknown_profile_returns_none(self):
        self.assertIsNone(
            get_user_profile("UNKNOWN", db_path=self.db_path)
        )

    def test_invalid_employment_type_is_rejected(self):
        with self.assertRaises(ValueError):
            save_user_profile(
                "TEST_USER", 25, 4, 35000, "Unknown",
                "Madhya Pradesh", db_path=self.db_path,
            )


if __name__ == "__main__":
    unittest.main()
