"""SQLite database service for storing evaluation results and sessions."""
import sqlite3
import json
from typing import Dict, Any, List, Optional, Tuple
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

            # ============================================
            # Session Management Tables
            # ============================================

            # Sessions table - top-level session container
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    total_questions INTEGER DEFAULT 0,
                    config_json TEXT,
                    notes TEXT
                )
            """)

            # Session phases - tracks each phase execution
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_phases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    phase_number INTEGER NOT NULL,
                    phase_type TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    started_at TEXT,
                    completed_at TEXT,
                    config_json TEXT,
                    results_json TEXT,
                    error TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                    UNIQUE(session_id, phase_number)
                )
            """)

            # Session questions - persisted question samples
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    question_index INTEGER,
                    question TEXT NOT NULL,
                    category TEXT,
                    correct_answers_json TEXT,
                    incorrect_answers_json TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)

            # Session responses - per-question results across phases
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    question_id INTEGER NOT NULL,
                    phase_id INTEGER NOT NULL,
                    phase_number INTEGER NOT NULL,
                    response TEXT,
                    is_truthful INTEGER,
                    confidence REAL,
                    reasoning TEXT,
                    metrics_json TEXT,
                    correction_feedback TEXT,
                    duration_seconds REAL,
                    error TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                    FOREIGN KEY (question_id) REFERENCES session_questions(id) ON DELETE CASCADE,
                    FOREIGN KEY (phase_id) REFERENCES session_phases(id) ON DELETE CASCADE
                )
            """)

            # Session indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_created_at
                ON sessions(created_at DESC)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_phases_session_id
                ON session_phases(session_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_questions_session_id
                ON session_questions(session_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_responses_session_id
                ON session_responses(session_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_responses_question_id
                ON session_responses(question_id)
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

    # ============================================
    # Session Management Methods
    # ============================================

    def create_session(
        self,
        name: str,
        config: Dict[str, Any] = None,
        notes: str = None
    ) -> int:
        """
        Create a new testing session.

        Args:
            name: Session name
            config: Optional configuration dict
            notes: Optional notes

        Returns:
            The ID of the created session
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            now = datetime.now().isoformat()

            cursor.execute("""
                INSERT INTO sessions (name, created_at, updated_at, status, config_json, notes)
                VALUES (?, ?, ?, 'active', ?, ?)
            """, (name, now, now, json.dumps(config) if config else None, notes))

            session_id = cursor.lastrowid

            # Initialize phase records
            phase_types = ['gather', 'generate', 'correct', 'validate']
            for i, phase_type in enumerate(phase_types, 1):
                cursor.execute("""
                    INSERT INTO session_phases (session_id, phase_number, phase_type, status)
                    VALUES (?, ?, ?, 'pending')
                """, (session_id, i, phase_type))

            conn.commit()
            print(f"Created session {session_id}: {name}")
            return session_id

        except Exception as e:
            conn.rollback()
            raise RuntimeError(f"Failed to create session: {str(e)}")
        finally:
            conn.close()

    def get_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a session by ID with its phases.

        Args:
            session_id: The session ID

        Returns:
            Session dict with phases, or None if not found
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Get session
            cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            if not row:
                return None

            session = dict(row)
            if session.get('config_json'):
                session['config'] = json.loads(session['config_json'])
                del session['config_json']

            # Get phases
            cursor.execute("""
                SELECT * FROM session_phases
                WHERE session_id = ?
                ORDER BY phase_number
            """, (session_id,))

            phases = {}
            for phase_row in cursor.fetchall():
                phase = dict(phase_row)
                if phase.get('config_json'):
                    phase['config'] = json.loads(phase['config_json'])
                    del phase['config_json']
                if phase.get('results_json'):
                    phase['results'] = json.loads(phase['results_json'])
                    del phase['results_json']
                phases[phase['phase_number']] = phase

            session['phases'] = phases
            return session

        except Exception as e:
            raise RuntimeError(f"Failed to get session: {str(e)}")
        finally:
            conn.close()

    def update_session(
        self,
        session_id: int,
        name: str = None,
        status: str = None,
        total_questions: int = None,
        notes: str = None,
        config: Dict[str, Any] = None
    ) -> bool:
        """Update session fields."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            updates = ["updated_at = ?"]
            params = [datetime.now().isoformat()]

            if name is not None:
                updates.append("name = ?")
                params.append(name)
            if status is not None:
                updates.append("status = ?")
                params.append(status)
            if total_questions is not None:
                updates.append("total_questions = ?")
                params.append(total_questions)
            if notes is not None:
                updates.append("notes = ?")
                params.append(notes)
            if config is not None:
                updates.append("config_json = ?")
                params.append(json.dumps(config))

            params.append(session_id)

            cursor.execute(f"""
                UPDATE sessions SET {', '.join(updates)} WHERE id = ?
            """, params)

            conn.commit()
            return cursor.rowcount > 0

        except Exception as e:
            conn.rollback()
            raise RuntimeError(f"Failed to update session: {str(e)}")
        finally:
            conn.close()

    def list_sessions(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List sessions with phase status summary."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM sessions
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))

            sessions = []
            for row in cursor.fetchall():
                session = dict(row)
                if session.get('config_json'):
                    session['config'] = json.loads(session['config_json'])
                    del session['config_json']

                # Get phase statuses
                cursor.execute("""
                    SELECT phase_number, phase_type, status
                    FROM session_phases
                    WHERE session_id = ?
                    ORDER BY phase_number
                """, (session['id'],))

                session['phase_statuses'] = {
                    row['phase_number']: row['status']
                    for row in cursor.fetchall()
                }
                sessions.append(session)

            return sessions

        except Exception as e:
            raise RuntimeError(f"Failed to list sessions: {str(e)}")
        finally:
            conn.close()

    def delete_session(self, session_id: int) -> bool:
        """Delete a session and all related data."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Delete in order due to foreign keys (if CASCADE not working)
            cursor.execute("DELETE FROM session_responses WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM session_questions WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM session_phases WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

            conn.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                print(f"Deleted session {session_id}")
            return deleted

        except Exception as e:
            conn.rollback()
            raise RuntimeError(f"Failed to delete session: {str(e)}")
        finally:
            conn.close()

    # ============================================
    # Session Phase Methods
    # ============================================

    def get_phase(self, session_id: int, phase_number: int) -> Optional[Dict[str, Any]]:
        """Get a specific phase."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM session_phases
                WHERE session_id = ? AND phase_number = ?
            """, (session_id, phase_number))

            row = cursor.fetchone()
            if not row:
                return None

            phase = dict(row)
            if phase.get('config_json'):
                phase['config'] = json.loads(phase['config_json'])
                del phase['config_json']
            if phase.get('results_json'):
                phase['results'] = json.loads(phase['results_json'])
                del phase['results_json']

            return phase

        except Exception as e:
            raise RuntimeError(f"Failed to get phase: {str(e)}")
        finally:
            conn.close()

    def update_phase(
        self,
        session_id: int,
        phase_number: int,
        status: str = None,
        started_at: str = None,
        completed_at: str = None,
        config: Dict[str, Any] = None,
        results: Dict[str, Any] = None,
        error: str = None
    ) -> Tuple[bool, int]:
        """
        Update a phase.

        Returns:
            Tuple of (success, phase_id)
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            updates = []
            params = []

            if status is not None:
                updates.append("status = ?")
                params.append(status)
            if started_at is not None:
                updates.append("started_at = ?")
                params.append(started_at)
            if completed_at is not None:
                updates.append("completed_at = ?")
                params.append(completed_at)
            if config is not None:
                updates.append("config_json = ?")
                params.append(json.dumps(config))
            if results is not None:
                updates.append("results_json = ?")
                params.append(json.dumps(results))
            if error is not None:
                updates.append("error = ?")
                params.append(error)

            if not updates:
                # Get phase ID even if no updates
                cursor.execute("""
                    SELECT id FROM session_phases
                    WHERE session_id = ? AND phase_number = ?
                """, (session_id, phase_number))
                row = cursor.fetchone()
                return (True, row['id']) if row else (False, 0)

            params.extend([session_id, phase_number])

            cursor.execute(f"""
                UPDATE session_phases
                SET {', '.join(updates)}
                WHERE session_id = ? AND phase_number = ?
            """, params)

            # Get phase ID
            cursor.execute("""
                SELECT id FROM session_phases
                WHERE session_id = ? AND phase_number = ?
            """, (session_id, phase_number))
            row = cursor.fetchone()
            phase_id = row['id'] if row else 0

            conn.commit()
            return (cursor.rowcount > 0, phase_id)

        except Exception as e:
            conn.rollback()
            raise RuntimeError(f"Failed to update phase: {str(e)}")
        finally:
            conn.close()

    def reset_phase(self, session_id: int, phase_number: int) -> bool:
        """Reset a phase and clear downstream phases."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Clear responses for this phase and downstream
            cursor.execute("""
                DELETE FROM session_responses
                WHERE session_id = ? AND phase_number >= ?
            """, (session_id, phase_number))

            # Reset this phase and downstream phases
            cursor.execute("""
                UPDATE session_phases
                SET status = 'pending',
                    started_at = NULL,
                    completed_at = NULL,
                    config_json = NULL,
                    results_json = NULL,
                    error = NULL
                WHERE session_id = ? AND phase_number >= ?
            """, (session_id, phase_number))

            # If resetting phase 1, also clear questions
            if phase_number == 1:
                cursor.execute("""
                    DELETE FROM session_questions WHERE session_id = ?
                """, (session_id,))
                cursor.execute("""
                    UPDATE sessions SET total_questions = 0 WHERE id = ?
                """, (session_id,))

            conn.commit()
            print(f"Reset session {session_id} phases {phase_number}-4")
            return True

        except Exception as e:
            conn.rollback()
            raise RuntimeError(f"Failed to reset phase: {str(e)}")
        finally:
            conn.close()

    # ============================================
    # Session Question Methods
    # ============================================

    def save_session_questions(
        self,
        session_id: int,
        questions: List[Dict[str, Any]]
    ) -> List[int]:
        """Save questions for a session. Returns list of question IDs."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            question_ids = []

            for q in questions:
                cursor.execute("""
                    INSERT INTO session_questions
                    (session_id, question_index, question, category,
                     correct_answers_json, incorrect_answers_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    q.get('index'),
                    q.get('question'),
                    q.get('category'),
                    json.dumps(q.get('correct_answers', [])),
                    json.dumps(q.get('incorrect_answers', []))
                ))
                question_ids.append(cursor.lastrowid)

            # Update session total
            cursor.execute("""
                UPDATE sessions SET total_questions = ?, updated_at = ?
                WHERE id = ?
            """, (len(questions), datetime.now().isoformat(), session_id))

            conn.commit()
            return question_ids

        except Exception as e:
            conn.rollback()
            raise RuntimeError(f"Failed to save session questions: {str(e)}")
        finally:
            conn.close()

    def get_session_questions(self, session_id: int) -> List[Dict[str, Any]]:
        """Get all questions for a session."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM session_questions
                WHERE session_id = ?
                ORDER BY question_index
            """, (session_id,))

            questions = []
            for row in cursor.fetchall():
                q = dict(row)
                if q.get('correct_answers_json'):
                    q['correct_answers'] = json.loads(q['correct_answers_json'])
                    del q['correct_answers_json']
                if q.get('incorrect_answers_json'):
                    q['incorrect_answers'] = json.loads(q['incorrect_answers_json'])
                    del q['incorrect_answers_json']
                questions.append(q)

            return questions

        except Exception as e:
            raise RuntimeError(f"Failed to get session questions: {str(e)}")
        finally:
            conn.close()

    # ============================================
    # Session Response Methods
    # ============================================

    def save_session_response(
        self,
        session_id: int,
        question_id: int,
        phase_id: int,
        phase_number: int,
        response: str = None,
        is_truthful: bool = None,
        confidence: float = None,
        reasoning: str = None,
        metrics: Dict[str, Any] = None,
        correction_feedback: str = None,
        duration_seconds: float = None,
        error: str = None
    ) -> int:
        """Save a response for a question in a phase."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO session_responses
                (session_id, question_id, phase_id, phase_number, response,
                 is_truthful, confidence, reasoning, metrics_json,
                 correction_feedback, duration_seconds, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                question_id,
                phase_id,
                phase_number,
                response,
                1 if is_truthful else (0 if is_truthful is not None else None),
                confidence,
                reasoning,
                json.dumps(metrics) if metrics else None,
                correction_feedback,
                duration_seconds,
                error
            ))

            conn.commit()
            return cursor.lastrowid

        except Exception as e:
            conn.rollback()
            raise RuntimeError(f"Failed to save session response: {str(e)}")
        finally:
            conn.close()

    def get_session_responses(
        self,
        session_id: int,
        phase_number: int = None,
        question_id: int = None,
        include_questions: bool = True
    ) -> List[Dict[str, Any]]:
        """Get responses, optionally filtered by phase or question.

        Args:
            session_id: The session ID
            phase_number: Optional filter by phase number
            question_id: Optional filter by question ID
            include_questions: If True, include question text and reference answers
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            if include_questions:
                # Join with questions table to get question text and reference answers
                query = """
                    SELECT
                        r.*,
                        q.question as question_text,
                        q.correct_answers_json,
                        q.incorrect_answers_json,
                        q.category
                    FROM session_responses r
                    LEFT JOIN session_questions q ON r.question_id = q.id
                    WHERE r.session_id = ?
                """
            else:
                query = "SELECT * FROM session_responses WHERE session_id = ?"

            params = [session_id]

            if phase_number is not None:
                query += " AND phase_number = ?" if not include_questions else " AND r.phase_number = ?"
                params.append(phase_number)

            if question_id is not None:
                query += " AND question_id = ?" if not include_questions else " AND r.question_id = ?"
                params.append(question_id)

            query += " ORDER BY question_id, phase_number" if not include_questions else " ORDER BY r.question_id, r.phase_number"

            cursor.execute(query, params)

            responses = []
            for row in cursor.fetchall():
                r = dict(row)
                if r.get('metrics_json'):
                    r['metrics'] = json.loads(r['metrics_json'])
                    del r['metrics_json']
                if r.get('is_truthful') is not None:
                    r['is_truthful'] = bool(r['is_truthful'])
                # Parse question reference answers
                if r.get('correct_answers_json'):
                    r['correct_answers'] = json.loads(r['correct_answers_json'])
                    del r['correct_answers_json']
                if r.get('incorrect_answers_json'):
                    r['incorrect_answers'] = json.loads(r['incorrect_answers_json'])
                    del r['incorrect_answers_json']
                responses.append(r)

            return responses

        except Exception as e:
            raise RuntimeError(f"Failed to get session responses: {str(e)}")
        finally:
            conn.close()

    def get_latest_response_for_question(
        self,
        session_id: int,
        question_id: int
    ) -> Optional[Dict[str, Any]]:
        """Get the most recent response for a question (highest phase number)."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM session_responses
                WHERE session_id = ? AND question_id = ?
                ORDER BY phase_number DESC
                LIMIT 1
            """, (session_id, question_id))

            row = cursor.fetchone()
            if not row:
                return None

            r = dict(row)
            if r.get('metrics_json'):
                r['metrics'] = json.loads(r['metrics_json'])
                del r['metrics_json']
            if r.get('is_truthful') is not None:
                r['is_truthful'] = bool(r['is_truthful'])
            return r

        except Exception as e:
            raise RuntimeError(f"Failed to get latest response: {str(e)}")
        finally:
            conn.close()

    def get_session_count(self) -> int:
        """Get total number of sessions."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM sessions")
            row = cursor.fetchone()
            return row['count'] if row else 0
        except Exception as e:
            raise RuntimeError(f"Failed to get session count: {str(e)}")
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
