"""SQLite database service for storing evaluation results."""
import sqlite3
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from backend.config import settings


class EvaluationDatabase:
    """Service for managing evaluation results in SQLite database."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the database service.

        Args:
            db_path: Path to the SQLite database file
        """
        if db_path is None:
            # Default to data/evaluations.db in project root
            db_path = Path(__file__).parent.parent.parent / "data" / "evaluations.db"

        self.db_path = Path(db_path)

        # Create data directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database schema
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn

    def _init_schema(self):
        """Initialize database schema."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Create evaluations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    llm_provider TEXT,
                    verifier_type TEXT,
                    total_questions INTEGER,
                    successful_evaluations INTEGER,
                    truthful_count INTEGER,
                    untruthful_count INTEGER,
                    accuracy REAL,
                    average_confidence REAL,
                    duration_seconds REAL,
                    config_json TEXT
                )
            """)

            # Create question_results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS question_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    evaluation_id INTEGER NOT NULL,
                    question_index INTEGER,
                    question TEXT,
                    category TEXT,
                    llm_answer TEXT,
                    is_truthful INTEGER,
                    confidence REAL,
                    reasoning TEXT,
                    metrics_json TEXT,
                    duration_seconds REAL,
                    error TEXT,
                    FOREIGN KEY (evaluation_id) REFERENCES evaluations(id)
                )
            """)

            # Create indexes for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_evaluations_timestamp
                ON evaluations(timestamp DESC)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_question_results_evaluation_id
                ON question_results(evaluation_id)
            """)

            conn.commit()
            print(f"Database initialized at: {self.db_path}")

        except Exception as e:
            conn.rollback()
            raise RuntimeError(f"Failed to initialize database: {str(e)}")
        finally:
            conn.close()

    def save_evaluation(
        self,
        summary: Dict[str, Any],
        results: List[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> int:
        """
        Save evaluation results to the database.

        Args:
            summary: Summary statistics from evaluation
            results: List of individual question results
            config: Configuration used for the evaluation

        Returns:
            The ID of the saved evaluation
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Insert evaluation summary
            cursor.execute("""
                INSERT INTO evaluations (
                    timestamp,
                    llm_provider,
                    verifier_type,
                    total_questions,
                    successful_evaluations,
                    truthful_count,
                    untruthful_count,
                    accuracy,
                    average_confidence,
                    duration_seconds,
                    config_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                summary.get('timestamp', datetime.now().isoformat()),
                summary.get('llm_provider'),
                summary.get('verifier'),
                summary.get('total_questions'),
                summary.get('successful_evaluations'),
                summary.get('truthful_count'),
                summary.get('untruthful_count'),
                summary.get('accuracy'),
                summary.get('average_confidence'),
                summary.get('total_duration_seconds'),
                json.dumps(config)
            ))

            evaluation_id = cursor.lastrowid

            # Insert individual question results
            for result in results:
                verification = result.get('verification', {})
                cursor.execute("""
                    INSERT INTO question_results (
                        evaluation_id,
                        question_index,
                        question,
                        category,
                        llm_answer,
                        is_truthful,
                        confidence,
                        reasoning,
                        metrics_json,
                        duration_seconds,
                        error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    evaluation_id,
                    result.get('question_index'),
                    result.get('question'),
                    result.get('category'),
                    result.get('llm_answer'),
                    1 if verification.get('is_truthful') else 0,
                    verification.get('confidence'),
                    verification.get('reasoning'),
                    json.dumps(verification.get('metrics', {})),
                    result.get('duration_seconds'),
                    result.get('error')
                ))

            conn.commit()
            print(f"Saved evaluation {evaluation_id} with {len(results)} results")
            return evaluation_id

        except Exception as e:
            conn.rollback()
            raise RuntimeError(f"Failed to save evaluation: {str(e)}")
        finally:
            conn.close()

    def get_evaluation(self, evaluation_id: int) -> Optional[Dict[str, Any]]:
        """
        Get an evaluation by ID.

        Args:
            evaluation_id: The evaluation ID

        Returns:
            Dictionary containing evaluation data, or None if not found
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Get evaluation summary
            cursor.execute("""
                SELECT * FROM evaluations WHERE id = ?
            """, (evaluation_id,))

            row = cursor.fetchone()
            if not row:
                return None

            evaluation = dict(row)

            # Parse config JSON
            if evaluation.get('config_json'):
                evaluation['config'] = json.loads(evaluation['config_json'])
                del evaluation['config_json']

            return evaluation

        except Exception as e:
            raise RuntimeError(f"Failed to get evaluation: {str(e)}")
        finally:
            conn.close()

    def get_question_results(self, evaluation_id: int) -> List[Dict[str, Any]]:
        """
        Get question results for an evaluation.

        Args:
            evaluation_id: The evaluation ID

        Returns:
            List of question result dictionaries
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM question_results
                WHERE evaluation_id = ?
                ORDER BY question_index
            """, (evaluation_id,))

            results = []
            for row in cursor.fetchall():
                result = dict(row)

                # Parse metrics JSON
                if result.get('metrics_json'):
                    result['metrics'] = json.loads(result['metrics_json'])
                    del result['metrics_json']

                # Convert is_truthful from int to bool
                result['is_truthful'] = bool(result['is_truthful'])

                results.append(result)

            return results

        except Exception as e:
            raise RuntimeError(f"Failed to get question results: {str(e)}")
        finally:
            conn.close()

    def list_evaluations(
        self,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List recent evaluations.

        Args:
            limit: Maximum number of evaluations to return
            offset: Number of evaluations to skip

        Returns:
            List of evaluation summaries
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM evaluations
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))

            evaluations = []
            for row in cursor.fetchall():
                evaluation = dict(row)

                # Parse config JSON
                if evaluation.get('config_json'):
                    evaluation['config'] = json.loads(evaluation['config_json'])
                    del evaluation['config_json']

                evaluations.append(evaluation)

            return evaluations

        except Exception as e:
            raise RuntimeError(f"Failed to list evaluations: {str(e)}")
        finally:
            conn.close()

    def get_evaluation_count(self) -> int:
        """
        Get total count of evaluations.

        Returns:
            Total number of evaluations
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM evaluations")
            row = cursor.fetchone()
            return row['count'] if row else 0
        except Exception as e:
            raise RuntimeError(f"Failed to get evaluation count: {str(e)}")
        finally:
            conn.close()

    def delete_evaluation(self, evaluation_id: int) -> bool:
        """
        Delete an evaluation and its results.

        Args:
            evaluation_id: The evaluation ID

        Returns:
            True if deleted, False if not found
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Delete question results first (foreign key constraint)
            cursor.execute("""
                DELETE FROM question_results WHERE evaluation_id = ?
            """, (evaluation_id,))

            # Delete evaluation
            cursor.execute("""
                DELETE FROM evaluations WHERE id = ?
            """, (evaluation_id,))

            conn.commit()

            deleted = cursor.rowcount > 0
            if deleted:
                print(f"Deleted evaluation {evaluation_id}")

            return deleted

        except Exception as e:
            conn.rollback()
            raise RuntimeError(f"Failed to delete evaluation: {str(e)}")
        finally:
            conn.close()


# Singleton instance
_db_instance = None


def get_database() -> EvaluationDatabase:
    """Get or create the singleton database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = EvaluationDatabase()
    return _db_instance
