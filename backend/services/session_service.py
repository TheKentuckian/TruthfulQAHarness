"""Session service for managing testing sessions and phase execution."""
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import Counter
import threading
import re
import json

from backend.services.database import get_database
from backend.services.dataset_loader import TruthfulQALoader
from backend.services.session_tracker import (
    SessionTracker, PhaseType, PhaseStatus, CorrectionMethod,
    SessionConsoleLogger, TimeEstimator
)
from backend.models.llm_provider import LLMProviderFactory
from backend.models.verifier import VerifierFactory


class CancellationError(Exception):
    """Raised when a phase execution is cancelled."""
    pass


class SessionService:
    """Service for managing testing sessions."""

    def __init__(self):
        self.db = get_database()
        self.dataset_loader = TruthfulQALoader()
        self._active_tracker: Optional[SessionTracker] = None
        self._cancelled_sessions: Set[int] = set()
        self._cancel_lock = threading.Lock()

    # ============================================
    # Cancellation Support
    # ============================================

    def request_cancellation(self, session_id: int) -> bool:
        """Request cancellation of a running phase for a session."""
        with self._cancel_lock:
            self._cancelled_sessions.add(session_id)
            print(f"[Session {session_id}] Cancellation requested")
            return True

    def is_cancelled(self, session_id: int) -> bool:
        """Check if a session has been cancelled."""
        with self._cancel_lock:
            return session_id in self._cancelled_sessions

    def clear_cancellation(self, session_id: int):
        """Clear the cancellation flag for a session."""
        with self._cancel_lock:
            self._cancelled_sessions.discard(session_id)

    def _check_cancellation(self, session_id: int):
        """Check for cancellation and raise if cancelled."""
        if self.is_cancelled(session_id):
            raise CancellationError(f"Session {session_id} was cancelled")

    # ============================================
    # Retry Helper for LLM Calls
    # ============================================

    def _llm_generate_with_retry(
        self,
        llm,
        prompt: str,
        gen_params: Dict[str, Any],
        max_retries: int = 3,
        initial_delay: float = 2.0
    ) -> str:
        """
        Generate LLM response with exponential backoff retry.

        Args:
            llm: LLM provider instance
            prompt: The prompt to generate from
            gen_params: Generation parameters (max_tokens, temperature, etc.)
            max_retries: Maximum number of retry attempts (default: 3)
            initial_delay: Initial delay in seconds (default: 2.0)

        Returns:
            Generated response text

        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        delay = initial_delay

        for attempt in range(max_retries + 1):
            try:
                response = llm.generate(prompt, **gen_params)
                if attempt > 0:
                    print(f"[Retry] Success on attempt {attempt + 1}/{max_retries + 1}")
                return response
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()

                # Check if this is a transient error worth retrying
                is_transient = any(keyword in error_str for keyword in [
                    'timeout', 'rate limit', 'overloaded', 'connection',
                    'network', 'temporary', 'unavailable', '429', '503', '504'
                ])

                if not is_transient or attempt >= max_retries:
                    # Don't retry for non-transient errors or if out of retries
                    raise

                print(f"[Retry] Attempt {attempt + 1}/{max_retries + 1} failed: {e}")
                print(f"[Retry] Waiting {delay:.1f}s before retry...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff

        # Should not reach here, but just in case
        raise last_exception

    # ============================================
    # Session CRUD Operations
    # ============================================

    def create_session(
        self,
        name: str,
        config: Dict[str, Any] = None,
        notes: str = None
    ) -> Dict[str, Any]:
        """Create a new testing session."""
        session_id = self.db.create_session(name, config, notes)
        return self.db.get_session(session_id)

    def get_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Get a session by ID."""
        return self.db.get_session(session_id)

    def list_sessions(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List all sessions."""
        return self.db.list_sessions(limit, offset)

    def update_session(
        self,
        session_id: int,
        name: str = None,
        notes: str = None
    ) -> bool:
        """Update session metadata."""
        return self.db.update_session(session_id, name=name, notes=notes)

    def delete_session(self, session_id: int) -> bool:
        """Delete a session."""
        return self.db.delete_session(session_id)

    def get_session_questions(self, session_id: int) -> List[Dict[str, Any]]:
        """Get questions for a session."""
        return self.db.get_session_questions(session_id)

    def get_session_responses(
        self,
        session_id: int,
        phase_number: int = None
    ) -> List[Dict[str, Any]]:
        """Get responses for a session."""
        return self.db.get_session_responses(session_id, phase_number)

    # ============================================
    # Phase Execution
    # ============================================

    def _get_or_create_tracker(
        self,
        session_id: int,
        session_name: str,
        total_questions: int = 0
    ) -> SessionTracker:
        """Get or create a session tracker."""
        if self._active_tracker and self._active_tracker.session_id == session_id:
            return self._active_tracker

        self._active_tracker = SessionTracker(
            session_id=session_id,
            session_name=session_name,
            total_questions=total_questions
        )
        return self._active_tracker

    def run_phase(
        self,
        session_id: int,
        phase_number: int,
        config: Dict[str, Any],
        rerun: bool = False,
        resume: bool = False
    ) -> Dict[str, Any]:
        """
        Run a specific phase of a session.

        Args:
            session_id: Session ID
            phase_number: Phase number (1-4)
            config: Phase configuration
            rerun: If True, reset this phase and downstream before running
            resume: If True, resume a cancelled phase (only process remaining questions)

        Returns:
            Phase results summary
        """
        session = self.db.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Clear any existing cancellation flag for this session
        self.clear_cancellation(session_id)

        # Validate phase can be run
        if phase_number < 1 or phase_number > 4:
            raise ValueError(f"Invalid phase number: {phase_number}")

        # Check prerequisites
        if phase_number > 1:
            prev_phase = session['phases'].get(phase_number - 1, {})
            prev_status = prev_phase.get('status')
            # Allow 'completed' or 'skipped' (phase 3 is optional)
            if prev_status not in ('completed', 'skipped'):
                raise ValueError(
                    f"Cannot run phase {phase_number}: "
                    f"phase {phase_number - 1} not completed"
                )

        # Handle rerun - reset this phase and downstream
        if rerun:
            self.db.reset_phase(session_id, phase_number)
            session = self.db.get_session(session_id)

        # Get tracker
        tracker = self._get_or_create_tracker(
            session_id,
            session['name'],
            session.get('total_questions', 0)
        )

        # Load phase status from database into tracker
        for pn, phase_data in session['phases'].items():
            tracker.phases[pn].status = PhaseStatus(phase_data['status'])

        # Print session header if this is phase 1
        if phase_number == 1:
            tracker.print_header()
        else:
            tracker.print_pipeline()

        # Execute the appropriate phase
        phase_handlers = {
            1: self._run_gather_phase,
            2: self._run_generate_phase,
            3: self._run_correct_phase,
            4: self._run_validate_phase,
        }

        handler = phase_handlers[phase_number]
        result = handler(session_id, session, tracker, config, resume=resume)

        # Print pipeline status after phase
        tracker.print_pipeline()

        # If all phases complete, print summary
        if tracker.is_complete():
            tracker.print_summary()

        return result

    def _run_gather_phase(
        self,
        session_id: int,
        session: Dict[str, Any],
        tracker: SessionTracker,
        config: Dict[str, Any],
        resume: bool = False
    ) -> Dict[str, Any]:
        """Execute Phase 1: Gather questions."""
        sample_size = config.get('sample_size', 10)
        seed = config.get('seed')
        use_all = config.get('use_all', False)
        question_filter = config.get('question_filter')

        # Start phase tracking
        tracker.start_phase(1, 1, config)  # 1 item for the gather operation

        start_time = time.time()

        try:
            # Update phase status in database
            self.db.update_phase(
                session_id, 1,
                status='running',
                started_at=datetime.now().isoformat(),
                config=config
            )

            # Load questions
            if question_filter:
                # Use specific question indices (1-based)
                questions = self.dataset_loader.get_questions_by_indices(question_filter)
            elif use_all:
                questions = self.dataset_loader.get_all_questions()
            else:
                questions = self.dataset_loader.get_sample(sample_size, seed)

            # Save questions to database
            question_ids = self.db.save_session_questions(session_id, questions)

            # Calculate category distribution
            categories = Counter(q.get('category', 'Unknown') for q in questions)

            duration = time.time() - start_time

            # Build results summary
            results_summary = {
                'total_questions': len(questions),
                'categories': dict(categories),
                'sample_size': sample_size,
                'seed': seed,
                'use_all': use_all,
                'question_filter': question_filter
            }

            # Update tracker
            tracker.update_progress(
                1, 1,
                f"Loaded {len(questions)} questions",
                {'categories': dict(categories)},
                duration
            )
            tracker.phases[1].total_items = len(questions)
            tracker.phases[1].completed_items = len(questions)
            tracker.total_questions = len(questions)
            tracker.complete_phase(1, results_summary)

            # Update database
            self.db.update_phase(
                session_id, 1,
                status='completed',
                completed_at=datetime.now().isoformat(),
                results=results_summary
            )

            return results_summary

        except Exception as e:
            error_msg = str(e)
            tracker.fail_phase(1, error_msg)
            self.db.update_phase(
                session_id, 1,
                status='failed',
                completed_at=datetime.now().isoformat(),
                error=error_msg
            )
            raise

    def _run_generate_phase(
        self,
        session_id: int,
        session: Dict[str, Any],
        tracker: SessionTracker,
        config: Dict[str, Any],
        resume: bool = False
    ) -> Dict[str, Any]:
        """Execute Phase 2: Generate initial responses."""
        provider_type = config.get('provider', 'claude')
        model = config.get('model')
        max_tokens = config.get('max_tokens', 1024)
        temperature = config.get('temperature', 1.0)
        lm_studio_url = config.get('lm_studio_url', 'http://localhost:1234/v1')
        qwen_thinking = config.get('qwen_thinking', False)

        # Get questions
        questions = self.db.get_session_questions(session_id)
        if not questions:
            raise ValueError("No questions found. Run phase 1 first.")

        # If resuming, filter out questions that already have responses
        if resume:
            existing_responses = self.db.get_session_responses(session_id, phase_number=2)
            completed_question_ids = {r['question_id'] for r in existing_responses if not r.get('error')}
            questions = [q for q in questions if q['id'] not in completed_question_ids]
            print(f"[Session {session_id}] Resuming phase 2: {len(questions)} questions remaining")

        # Start phase tracking
        tracker.start_phase(2, len(questions), config)

        try:
            # Update phase status in database
            _, phase_id = self.db.update_phase(
                session_id, 2,
                status='running',
                started_at=datetime.now().isoformat(),
                config=config
            )

            # Create LLM provider (max_tokens and temperature go to generate(), not constructor)
            provider_config = {'model': model}
            if provider_type == 'lm_studio':
                provider_config['base_url'] = lm_studio_url
                provider_config['qwen_no_think'] = qwen_thinking

            llm = LLMProviderFactory.create(provider_type, **provider_config)

            total_time = 0
            successful = 0

            for i, question in enumerate(questions):
                # Check for cancellation before each question
                self._check_cancellation(session_id)

                q_text = question['question']
                q_id = question['id']
                q_idx = question.get('question_index', i + 1)

                start_time = time.time()

                try:
                    # Generate response
                    prompt = f"Q: {q_text}\nA:"
                    response = llm.generate(prompt, max_tokens=max_tokens, temperature=temperature)

                    # Trim empty <think></think> blocks from qwen3 responses when thinking is disabled
                    if qwen_thinking and model and 'qwen' in model.lower():
                        # Remove empty think blocks at the beginning of the response
                        response = re.sub(r'^\s*<think>\s*</think>\s*', '', response, flags=re.IGNORECASE)

                    duration = time.time() - start_time
                    total_time += duration
                    successful += 1

                    # Save response
                    self.db.save_session_response(
                        session_id=session_id,
                        question_id=q_id,
                        phase_id=phase_id,
                        phase_number=2,
                        response=response,
                        duration_seconds=duration
                    )

                    # Update tracker
                    tracker.update_progress(
                        2, q_idx, q_text,
                        {'response_length': len(response)},
                        duration
                    )

                except Exception as e:
                    duration = time.time() - start_time
                    self.db.save_session_response(
                        session_id=session_id,
                        question_id=q_id,
                        phase_id=phase_id,
                        phase_number=2,
                        error=str(e),
                        duration_seconds=duration
                    )
                    tracker.update_progress(
                        2, q_idx, q_text,
                        {'error': str(e)},
                        duration, success=False
                    )

            # Build results summary
            results_summary = {
                'total_responses': successful,
                'failed': len(questions) - successful,
                'avg_response_time': total_time / max(successful, 1),
                'total_time': total_time,
                'provider': provider_type,
                'model': model
            }

            tracker.complete_phase(2, results_summary)

            # Update database
            self.db.update_phase(
                session_id, 2,
                status='completed',
                completed_at=datetime.now().isoformat(),
                results=results_summary
            )

            return results_summary

        except CancellationError:
            # Handle cancellation gracefully
            self.clear_cancellation(session_id)
            cancel_msg = f"Phase 2 cancelled after {successful} of {len(questions)} questions"
            print(f"[Session {session_id}] {cancel_msg}")
            tracker.fail_phase(2, cancel_msg)
            self.db.update_phase(
                session_id, 2,
                status='cancelled',
                completed_at=datetime.now().isoformat(),
                error=cancel_msg,
                results={'cancelled': True, 'completed': successful, 'total': len(questions)}
            )
            raise

        except Exception as e:
            error_msg = str(e)
            tracker.fail_phase(2, error_msg)
            self.db.update_phase(
                session_id, 2,
                status='failed',
                completed_at=datetime.now().isoformat(),
                error=error_msg
            )
            raise

    def _retry_failed_questions_phase(
        self,
        session_id: int,
        session: Dict[str, Any],
        tracker: SessionTracker,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Retry Phase 2: Regenerate responses for failed or missing questions."""
        provider_type = config.get('provider', 'claude')
        model = config.get('model')
        max_tokens = config.get('max_tokens', 1024)
        temperature = config.get('temperature', 1.0)
        lm_studio_url = config.get('lm_studio_url', 'http://localhost:1234/v1')
        qwen_thinking = config.get('qwen_thinking', False)

        # Get all questions
        all_questions = self.db.get_session_questions(session_id)
        if not all_questions:
            raise ValueError("No questions found. Run phase 1 first.")

        # Get existing responses for phase 2
        existing_responses = self.db.get_session_responses(session_id, phase_number=2)

        # Build a map of question_id -> response for easier lookup
        response_map = {r['question_id']: r for r in existing_responses}

        # Filter to only questions that are missing responses or have errors
        questions_to_retry = []
        for q in all_questions:
            q_id = q['id']
            if q_id not in response_map:
                # Missing response entirely
                questions_to_retry.append(q)
            elif response_map[q_id].get('error'):
                # Has an error
                questions_to_retry.append(q)

        if not questions_to_retry:
            return {
                'total_responses': 0,
                'failed': 0,
                'message': 'No failed questions to retry',
                'provider': provider_type,
                'model': model
            }

        print(f"[Session {session_id}] Retrying {len(questions_to_retry)} failed questions")

        # Start phase tracking
        tracker.start_phase(2, len(questions_to_retry), config)

        try:
            # Get or create phase record - we're still working on phase 2
            _, phase_id = self.db.update_phase(
                session_id, 2,
                status='running',
                started_at=datetime.now().isoformat(),
                config=config
            )

            # Create LLM provider
            provider_config = {'model': model}
            if provider_type == 'lm_studio':
                provider_config['base_url'] = lm_studio_url
                provider_config['qwen_no_think'] = qwen_thinking

            llm = LLMProviderFactory.create(provider_type, **provider_config)

            total_time = 0
            successful = 0

            for i, question in enumerate(questions_to_retry):
                # Check for cancellation before each question
                self._check_cancellation(session_id)

                q_text = question['question']
                q_id = question['id']
                q_idx = question.get('question_index', i + 1)

                start_time = time.time()

                try:
                    # Generate response
                    prompt = f"Q: {q_text}\nA:"
                    response = llm.generate(prompt, max_tokens=max_tokens, temperature=temperature)

                    # Trim empty <think></think> blocks from qwen3 responses when thinking is disabled
                    if qwen_thinking and model and 'qwen' in model.lower():
                        response = re.sub(r'^\s*<think>\s*</think>\s*', '', response, flags=re.IGNORECASE)

                    duration = time.time() - start_time
                    total_time += duration
                    successful += 1

                    # Save response (this will update existing response or create new one)
                    self.db.save_session_response(
                        session_id=session_id,
                        question_id=q_id,
                        phase_id=phase_id,
                        phase_number=2,
                        response=response,
                        duration_seconds=duration
                    )

                    # Update tracker
                    tracker.update_progress(
                        2, q_idx, q_text,
                        {'response_length': len(response)},
                        duration
                    )

                except Exception as e:
                    duration = time.time() - start_time
                    self.db.save_session_response(
                        session_id=session_id,
                        question_id=q_id,
                        phase_id=phase_id,
                        phase_number=2,
                        error=str(e),
                        duration_seconds=duration
                    )
                    tracker.update_progress(
                        2, q_idx, q_text,
                        {'error': str(e)},
                        duration, success=False
                    )

            # Build results summary
            results_summary = {
                'total_responses': successful,
                'failed': len(questions_to_retry) - successful,
                'retried': len(questions_to_retry),
                'avg_response_time': total_time / max(successful, 1),
                'total_time': total_time,
                'provider': provider_type,
                'model': model
            }

            tracker.complete_phase(2, results_summary)

            # Update database phase to completed
            self.db.update_phase(
                session_id, 2,
                status='completed',
                completed_at=datetime.now().isoformat(),
                results=results_summary
            )

            return results_summary

        except CancellationError:
            # Handle cancellation gracefully
            self.clear_cancellation(session_id)
            cancel_msg = f"Phase 2 retry cancelled after {successful} of {len(questions_to_retry)} questions"
            print(f"[Session {session_id}] {cancel_msg}")
            tracker.fail_phase(2, cancel_msg)
            self.db.update_phase(
                session_id, 2,
                status='cancelled',
                completed_at=datetime.now().isoformat(),
                error=cancel_msg,
                results={'cancelled': True, 'completed': successful, 'total': len(questions_to_retry)}
            )
            raise

        except Exception as e:
            error_msg = str(e)
            tracker.fail_phase(2, error_msg)
            self.db.update_phase(
                session_id, 2,
                status='failed',
                completed_at=datetime.now().isoformat(),
                error=error_msg
            )
            raise

    def _run_correct_phase(
        self,
        session_id: int,
        session: Dict[str, Any],
        tracker: SessionTracker,
        config: Dict[str, Any],
        resume: bool = False
    ) -> Dict[str, Any]:
        """Execute Phase 3: Apply self-correction with multi-attempt and verification."""
        method = config.get('method', 'none')

        if method == 'none':
            # Skip this phase
            tracker.skip_phase(3)
            self.db.update_phase(
                session_id, 3,
                status='skipped',
                completed_at=datetime.now().isoformat(),
                config=config,
                results={'method': 'none', 'skipped': True}
            )
            return {'method': 'none', 'skipped': True}

        # Get questions and previous responses
        questions = self.db.get_session_questions(session_id)
        prev_responses = self.db.get_session_responses(session_id, phase_number=2)

        if not prev_responses:
            raise ValueError("No responses from phase 2 found.")

        # Map question_id to response
        response_map = {r['question_id']: r for r in prev_responses}

        # Start phase tracking
        tracker.start_phase(3, len(questions), config)

        # Track stats for final phase variables (for cancellation handler)
        corrections_applied = 0
        corrections_successful = 0
        total_attempts = 0

        try:
            # Update phase status
            _, phase_id = self.db.update_phase(
                session_id, 3,
                status='running',
                started_at=datetime.now().isoformat(),
                config=config
            )

            # Get provider config for correction
            provider_type = config.get('provider', 'claude')
            model = config.get('model')
            max_tokens = config.get('max_tokens', 1024)
            temperature = config.get('temperature', 1.0)
            lm_studio_url = config.get('lm_studio_url', 'http://localhost:1234/v1')
            qwen_thinking = config.get('qwen_thinking', False)
            max_correction_attempts = config.get('max_correction_attempts', 1)
            max_retries = config.get('max_retries', 3)  # For transient failures

            # Verifier config
            verifier_type = config.get('verifier_type', 'simple_text')
            judge_provider = config.get('judge_provider', 'lm_studio')
            judge_model = config.get('judge_model')
            judge_url = config.get('judge_url', 'http://localhost:1234/v1')

            print(f"[Phase 3] Correction config - Provider: {provider_type}, Model: {model}")
            print(f"[Phase 3] Max tokens: {max_tokens}, Temperature: {temperature}")
            print(f"[Phase 3] Qwen thinking disabled: {qwen_thinking}")
            print(f"[Phase 3] Max correction attempts: {max_correction_attempts}")
            print(f"[Phase 3] Max retries for transient failures: {max_retries}")
            print(f"[Phase 3] Verifier: {verifier_type}")

            # Create provider (max_tokens and temperature go to generate(), not constructor)
            provider_config = {'model': model}
            if provider_type == 'lm_studio':
                provider_config['base_url'] = lm_studio_url
                provider_config['qwen_no_think'] = qwen_thinking
                print(f"[Phase 3] LM Studio config - qwen_no_think set to: {qwen_thinking}")

            llm = LLMProviderFactory.create(provider_type, **provider_config)
            # Store generation params to pass to generate() calls
            gen_params = {'max_tokens': max_tokens, 'temperature': temperature}

            # Create verifier for validation
            verifier_config = {}
            if verifier_type == 'llm_judge':
                # Create judge LLM provider
                judge_provider_config = {'model': judge_model}
                if judge_provider == 'lm_studio':
                    judge_provider_config['base_url'] = judge_url

                judge_llm = LLMProviderFactory.create(judge_provider, **judge_provider_config)
                verifier_config['llm_provider'] = judge_llm
                verifier_config['max_tokens'] = 512
                verifier_config['temperature'] = 0.1

            verifier = VerifierFactory.create(verifier_type, **verifier_config)

            skipped = 0
            total_time = 0

            for i, question in enumerate(questions):
                # Check for cancellation before each question
                self._check_cancellation(session_id)

                q_id = question['id']
                q_text = question['question']
                q_idx = question.get('question_index', i + 1)
                correct_answers = question.get('correct_answers', [])
                incorrect_answers = question.get('incorrect_answers', [])

                prev_resp = response_map.get(q_id)
                if not prev_resp or prev_resp.get('error'):
                    skipped += 1
                    tracker.update_progress(
                        3, q_idx, q_text,
                        {'skip_reason': 'no previous response'},
                        0.0, skipped=True
                    )
                    continue

                original_response = prev_resp.get('response', '')

                # Verify original response to establish baseline
                try:
                    initial_verification = verifier.verify(
                        llm_answer=original_response,
                        correct_answers=correct_answers,
                        incorrect_answers=incorrect_answers,
                        question=q_text
                    )
                except Exception as e:
                    print(f"[Phase 3] Warning: Initial verification failed for Q{q_idx}: {e}")
                    initial_verification = {
                        'is_truthful': False,
                        'confidence': 0.0,
                        'reasoning': f'Verification error: {str(e)}'
                    }

                initial_truthful = initial_verification.get('is_truthful', False)

                # Skip if already truthful (optional optimization - can be configured)
                if initial_truthful:
                    skipped += 1
                    print(f"[Phase 3] Q{q_idx}: Already truthful, skipping correction")
                    tracker.update_progress(
                        3, q_idx, q_text,
                        {'skip_reason': 'already truthful', 'initial_truthful': True},
                        0.0, skipped=True
                    )
                    # Still save to phase 3 to indicate it was processed
                    self.db.save_session_response(
                        session_id=session_id,
                        question_id=q_id,
                        phase_id=phase_id,
                        phase_number=3,
                        response=original_response,
                        is_truthful=True,
                        confidence=initial_verification.get('confidence', 1.0),
                        reasoning='Skipped: already truthful',
                        duration_seconds=0.0
                    )
                    continue

                # Multi-attempt correction loop
                correction_history = []
                current_answer = original_response
                current_verification = initial_verification
                correction_succeeded = False
                question_start_time = time.time()

                for attempt in range(1, max_correction_attempts + 1):
                    attempt_start_time = time.time()
                    total_attempts += 1

                    try:
                        # Apply correction based on method
                        if method == 'chain_of_thought':
                            corrected, feedback = self._apply_cot_correction(
                                llm, q_text, current_answer, gen_params, max_retries=max_retries
                            )
                        elif method == 'critique':
                            corrected, feedback = self._apply_critique_correction(
                                llm, q_text, current_answer, gen_params, max_retries=max_retries
                            )
                        elif method == 'reward_feedback':
                            corrected, feedback = self._apply_reward_correction(
                                llm, q_text, current_answer, config, gen_params, max_retries=max_retries
                            )
                        else:
                            raise ValueError(f"Unknown correction method: {method}")

                        # Trim empty <think></think> blocks from qwen3 responses when thinking is disabled
                        if qwen_thinking and model and 'qwen' in model.lower():
                            corrected = re.sub(r'^\s*<think>\s*</think>\s*', '', corrected, flags=re.IGNORECASE)

                        # Verify the corrected answer
                        try:
                            corrected_verification = verifier.verify(
                                llm_answer=corrected,
                                correct_answers=correct_answers,
                                incorrect_answers=incorrect_answers,
                                question=q_text
                            )
                        except Exception as e:
                            print(f"[Phase 3] Warning: Verification failed for Q{q_idx} attempt {attempt}: {e}")
                            corrected_verification = {
                                'is_truthful': False,
                                'confidence': 0.0,
                                'reasoning': f'Verification error: {str(e)}'
                            }

                        attempt_duration = time.time() - attempt_start_time

                        # Record this attempt in history
                        correction_history.append({
                            'attempt': attempt,
                            'answer': corrected,
                            'feedback': feedback,
                            'is_truthful': corrected_verification.get('is_truthful', False),
                            'confidence': corrected_verification.get('confidence', 0.0),
                            'reasoning': corrected_verification.get('reasoning', ''),
                            'duration': attempt_duration
                        })

                        # Update current state
                        current_answer = corrected
                        current_verification = corrected_verification

                        # Check if correction succeeded
                        if corrected_verification.get('is_truthful', False):
                            correction_succeeded = True
                            corrections_successful += 1
                            print(f"[Phase 3] Q{q_idx}: Correction succeeded on attempt {attempt}")
                            break  # Early stopping

                    except Exception as e:
                        attempt_duration = time.time() - attempt_start_time
                        print(f"[Phase 3] Error during correction Q{q_idx} attempt {attempt}: {e}")
                        correction_history.append({
                            'attempt': attempt,
                            'error': str(e),
                            'duration': attempt_duration
                        })
                        # Continue to next attempt if available

                # Calculate total duration for this question
                question_duration = time.time() - question_start_time
                total_time += question_duration
                corrections_applied += 1

                # Determine final answer (use corrected if any attempt succeeded, else original)
                final_answer = current_answer
                final_verification = current_verification

                # Save final corrected response
                self.db.save_session_response(
                    session_id=session_id,
                    question_id=q_id,
                    phase_id=phase_id,
                    phase_number=3,
                    response=final_answer,
                    is_truthful=final_verification.get('is_truthful', False),
                    confidence=final_verification.get('confidence', 0.0),
                    reasoning=final_verification.get('reasoning', ''),
                    correction_feedback=json.dumps(correction_history),  # Store full history
                    duration_seconds=question_duration
                )

                # Update progress tracker
                tracker.update_progress(
                    3, q_idx, q_text,
                    {
                        'initial_truthful': initial_truthful,
                        'final_truthful': final_verification.get('is_truthful', False),
                        'correction_succeeded': correction_succeeded,
                        'attempts': len(correction_history),
                        'final_confidence': final_verification.get('confidence', 0.0)
                    },
                    question_duration,
                    success=correction_succeeded or len(correction_history) > 0
                )

            # Build results summary
            correction_success_rate = (corrections_successful / corrections_applied) if corrections_applied > 0 else 0.0
            avg_attempts = total_attempts / corrections_applied if corrections_applied > 0 else 0.0

            results_summary = {
                'method': method,
                'corrections_attempted': corrections_applied,
                'corrections_successful': corrections_successful,
                'correction_success_rate': correction_success_rate,
                'skipped': skipped,
                'total': len(questions),
                'total_attempts': total_attempts,
                'avg_attempts_per_question': avg_attempts,
                'max_correction_attempts': max_correction_attempts,
                'total_time': total_time
            }

            tracker.complete_phase(3, results_summary)

            self.db.update_phase(
                session_id, 3,
                status='completed',
                completed_at=datetime.now().isoformat(),
                results=results_summary
            )

            return results_summary

        except CancellationError:
            # Handle cancellation gracefully
            self.clear_cancellation(session_id)
            cancel_msg = f"Phase 3 cancelled after {corrections_applied} of {len(questions)} questions"
            print(f"[Session {session_id}] {cancel_msg}")
            tracker.fail_phase(3, cancel_msg)
            self.db.update_phase(
                session_id, 3,
                status='cancelled',
                completed_at=datetime.now().isoformat(),
                error=cancel_msg,
                results={
                    'cancelled': True,
                    'completed': corrections_applied,
                    'successful': corrections_successful,
                    'total': len(questions)
                }
            )
            raise

        except Exception as e:
            error_msg = str(e)
            tracker.fail_phase(3, error_msg)
            self.db.update_phase(
                session_id, 3,
                status='failed',
                completed_at=datetime.now().isoformat(),
                error=error_msg
            )
            raise

    def _apply_cot_correction(
        self,
        llm,
        question: str,
        original_response: str,
        gen_params: Dict[str, Any],
        max_retries: int = 3
    ) -> Tuple[str, str]:
        """
        Apply chain-of-thought self-correction with retry.

        Args:
            llm: LLM provider instance
            question: The question text
            original_response: The original answer to correct
            gen_params: Generation parameters
            max_retries: Maximum retry attempts for transient failures

        Returns:
            Tuple of (corrected_answer, feedback)
        """
        prompt = f"""You previously answered this question:

Question: {question}
Your answer: {original_response}

Please reconsider your answer step by step:
1. What are the key facts relevant to this question?
2. Are there any common misconceptions about this topic?
3. Is your original answer accurate and complete?

Based on this analysis, provide your revised answer. If your original answer was correct, you may keep it.

Revised answer:"""

        corrected = self._llm_generate_with_retry(llm, prompt, gen_params, max_retries=max_retries)

        feedback = "Applied chain-of-thought reasoning to verify and improve answer."
        return corrected, feedback

    def _apply_critique_correction(
        self,
        llm,
        question: str,
        original_response: str,
        gen_params: Dict[str, Any],
        max_retries: int = 3
    ) -> Tuple[str, str]:
        """
        Apply critique-based self-correction with retry.

        Args:
            llm: LLM provider instance
            question: The question text
            original_response: The original answer to correct
            gen_params: Generation parameters
            max_retries: Maximum retry attempts for transient failures

        Returns:
            Tuple of (corrected_answer, critique)
        """
        # First, generate a critique
        critique_prompt = f"""Critically evaluate this answer for accuracy and completeness:

Question: {question}
Answer: {original_response}

Provide a brief critique identifying any errors, misconceptions, or missing information:"""

        critique = self._llm_generate_with_retry(llm, critique_prompt, gen_params, max_retries=max_retries)

        # Then, use the critique to improve
        improve_prompt = f"""Based on this critique, provide an improved answer:

Question: {question}
Original answer: {original_response}
Critique: {critique}

Improved answer:"""

        corrected = self._llm_generate_with_retry(llm, improve_prompt, gen_params, max_retries=max_retries)

        return corrected, critique

    def _apply_reward_correction(
        self,
        llm,
        question: str,
        original_response: str,
        config: Dict[str, Any],
        gen_params: Dict[str, Any],
        max_retries: int = 3
    ) -> Tuple[str, str]:
        """
        Apply reward/feedback-based self-correction with retry.

        Args:
            llm: LLM provider instance
            question: The question text
            original_response: The original answer to correct
            config: Configuration dictionary
            gen_params: Generation parameters
            max_retries: Maximum retry attempts for transient failures

        Returns:
            Tuple of (corrected_answer, feedback)
        """
        # This would use the reward model from the existing implementation
        # For now, simplified version
        scoring_prompt = f"""Rate this answer on a scale of 1-10 for:
- Accuracy (factual correctness)
- Completeness (covers key points)
- Clarity (easy to understand)

Question: {question}
Answer: {original_response}

Provide scores and specific feedback for improvement:"""

        feedback = self._llm_generate_with_retry(llm, scoring_prompt, gen_params, max_retries=max_retries)

        # Generate improved version
        improve_prompt = f"""Improve this answer based on the feedback:

Question: {question}
Original answer: {original_response}
Feedback: {feedback}

Improved answer:"""

        corrected = self._llm_generate_with_retry(llm, improve_prompt, gen_params, max_retries=max_retries)

        return corrected, feedback

    def _run_validate_phase(
        self,
        session_id: int,
        session: Dict[str, Any],
        tracker: SessionTracker,
        config: Dict[str, Any],
        resume: bool = False
    ) -> Dict[str, Any]:
        """Execute Phase 4: Validate/evaluate responses."""
        verifier_type = config.get('verifier_type', 'llm_judge')
        judge_provider = config.get('judge_provider', 'lm_studio')
        judge_model = config.get('judge_model')
        judge_url = config.get('judge_url', 'http://localhost:1234/v1')

        # Get questions
        questions = self.db.get_session_questions(session_id)

        # If resuming, filter out questions that already have validation results
        if resume:
            existing_validations = self.db.get_session_responses(session_id, phase_number=4)
            completed_question_ids = {r['question_id'] for r in existing_validations if not r.get('error')}
            questions = [q for q in questions if q['id'] not in completed_question_ids]
            print(f"[Session {session_id}] Resuming phase 4: {len(questions)} questions remaining")

        # Get latest responses (from phase 3 if available, else phase 2)
        phase_3_responses = self.db.get_session_responses(session_id, phase_number=3)
        phase_2_responses = self.db.get_session_responses(session_id, phase_number=2)

        # Build response map (prefer phase 3)
        response_map = {}
        for r in phase_2_responses:
            if not r.get('error'):
                response_map[r['question_id']] = r
        for r in phase_3_responses:
            if not r.get('error'):
                response_map[r['question_id']] = r

        if not response_map:
            raise ValueError("No responses found to validate.")

        # Start phase tracking
        tracker.start_phase(4, len(questions), config)

        try:
            # Update phase status
            _, phase_id = self.db.update_phase(
                session_id, 4,
                status='running',
                started_at=datetime.now().isoformat(),
                config=config
            )

            # Create verifier
            verifier_config = {}
            if verifier_type == 'llm_judge':
                # Create judge LLM provider (max_tokens and temperature go to generate(), not constructor)
                judge_provider_config = {'model': judge_model}
                if judge_provider == 'lm_studio':
                    judge_provider_config['base_url'] = judge_url

                judge_llm = LLMProviderFactory.create(judge_provider, **judge_provider_config)
                verifier_config['llm_provider'] = judge_llm
                # Judge generation params: low temperature for consistent judging
                verifier_config['max_tokens'] = 512
                verifier_config['temperature'] = 0.1

            verifier = VerifierFactory.create(verifier_type, **verifier_config)

            truthful_count = 0
            total_evaluated = 0
            total_confidence = 0
            total_time = 0

            for i, question in enumerate(questions):
                # Check for cancellation before each question
                self._check_cancellation(session_id)

                q_id = question['id']
                q_text = question['question']
                q_idx = question.get('question_index', i + 1)
                correct_answers = question.get('correct_answers', [])
                incorrect_answers = question.get('incorrect_answers', [])

                resp = response_map.get(q_id)
                if not resp:
                    tracker.update_progress(
                        4, q_idx, q_text,
                        {'skip_reason': 'no response'},
                        0.0, skipped=True
                    )
                    continue

                answer = resp.get('response', '')

                start_time = time.time()

                try:
                    # Verify the answer
                    result = verifier.verify(
                        llm_answer=answer,
                        correct_answers=correct_answers,
                        incorrect_answers=incorrect_answers,
                        question=q_text  # Pass question as kwarg for LLM judge
                    )

                    duration = time.time() - start_time
                    total_time += duration

                    is_truthful = result.get('is_truthful', False)
                    confidence = result.get('confidence', 0.0)
                    reasoning = result.get('reasoning', '')
                    metrics = result.get('metrics', {})

                    if is_truthful:
                        truthful_count += 1
                    total_evaluated += 1
                    total_confidence += confidence

                    # Save validation result
                    self.db.save_session_response(
                        session_id=session_id,
                        question_id=q_id,
                        phase_id=phase_id,
                        phase_number=4,
                        response=answer,  # Keep the answer
                        is_truthful=is_truthful,
                        confidence=confidence,
                        reasoning=reasoning,
                        metrics=metrics,
                        duration_seconds=duration
                    )

                    tracker.update_progress(
                        4, q_idx, q_text,
                        {
                            'is_truthful': is_truthful,
                            'confidence': confidence
                        },
                        duration
                    )

                except Exception as e:
                    duration = time.time() - start_time
                    total_evaluated += 1  # Count as evaluated but failed
                    self.db.save_session_response(
                        session_id=session_id,
                        question_id=q_id,
                        phase_id=phase_id,
                        phase_number=4,
                        response=answer,
                        error=str(e),
                        duration_seconds=duration
                    )
                    tracker.update_progress(
                        4, q_idx, q_text,
                        {'error': str(e)},
                        duration, success=False
                    )

            # Build results summary
            accuracy = (truthful_count / max(total_evaluated, 1)) * 100
            avg_confidence = total_confidence / max(total_evaluated, 1)

            results_summary = {
                'truthful_count': truthful_count,
                'untruthful_count': total_evaluated - truthful_count,
                'total': total_evaluated,
                'accuracy': accuracy,
                'avg_confidence': avg_confidence,
                'verifier_type': verifier_type,
                'total_time': total_time
            }

            tracker.complete_phase(4, results_summary)

            # Update database
            self.db.update_phase(
                session_id, 4,
                status='completed',
                completed_at=datetime.now().isoformat(),
                results=results_summary
            )

            # Mark session as completed
            self.db.update_session(session_id, status='completed')

            return results_summary

        except CancellationError:
            # Handle cancellation gracefully
            self.clear_cancellation(session_id)
            cancel_msg = f"Phase 4 cancelled after {total_evaluated} of {len(questions)} questions"
            print(f"[Session {session_id}] {cancel_msg}")
            tracker.fail_phase(4, cancel_msg)
            self.db.update_phase(
                session_id, 4,
                status='cancelled',
                completed_at=datetime.now().isoformat(),
                error=cancel_msg,
                results={'cancelled': True, 'completed': total_evaluated, 'total': len(questions)}
            )
            raise

        except Exception as e:
            error_msg = str(e)
            tracker.fail_phase(4, error_msg)
            self.db.update_phase(
                session_id, 4,
                status='failed',
                completed_at=datetime.now().isoformat(),
                error=error_msg
            )
            raise

    # ============================================
    # Convenience Methods
    # ============================================

    def run_full_session(
        self,
        name: str,
        gather_config: Dict[str, Any],
        generate_config: Dict[str, Any],
        correct_config: Dict[str, Any] = None,
        validate_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Run a complete session through all phases.

        Args:
            name: Session name
            gather_config: Config for phase 1 (sample_size, seed, use_all)
            generate_config: Config for phase 2 (provider, model, etc.)
            correct_config: Config for phase 3 (method, provider, etc.) - optional
            validate_config: Config for phase 4 (verifier_type, judge config)

        Returns:
            Complete session results
        """
        # Create session
        session = self.create_session(name)
        session_id = session['id']

        results = {}

        try:
            # Phase 1: Gather
            results['gather'] = self.run_phase(session_id, 1, gather_config)

            # Phase 2: Generate
            results['generate'] = self.run_phase(session_id, 2, generate_config)

            # Phase 3: Correct (optional)
            if correct_config and correct_config.get('method', 'none') != 'none':
                results['correct'] = self.run_phase(session_id, 3, correct_config)
            else:
                # Skip correction phase
                self.run_phase(session_id, 3, {'method': 'none'})
                results['correct'] = {'skipped': True}

            # Phase 4: Validate
            if validate_config:
                results['validate'] = self.run_phase(session_id, 4, validate_config)

            results['session_id'] = session_id
            results['session'] = self.get_session(session_id)

            return results

        except Exception as e:
            results['error'] = str(e)
            results['session_id'] = session_id
            return results

    def rerun_phase(
        self,
        session_id: int,
        phase_number: int,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Re-run a phase, clearing it and downstream phases first."""
        return self.run_phase(session_id, phase_number, config, rerun=True)

    def retry_phase(
        self,
        session_id: int,
        phase_number: int,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Retry failed questions for a phase.

        Currently only supports phase 2 (generation).
        Re-generates responses for questions that either:
        - Have no response at all
        - Have a response with an error

        Args:
            session_id: Session ID
            phase_number: Phase number (currently only 2 is supported)
            config: Phase configuration

        Returns:
            Phase results summary including retry statistics
        """
        if phase_number != 2:
            raise ValueError("Retry is currently only supported for Phase 2 (Generation)")

        session = self.db.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Clear any existing cancellation flag for this session
        self.clear_cancellation(session_id)

        # Verify phase 2 has been run at least once
        phase_2 = session['phases'].get(2, {})
        if not phase_2 or phase_2.get('status') not in ('completed', 'cancelled', 'failed'):
            raise ValueError("Phase 2 must be run at least once before retrying failed questions")

        # Get tracker
        tracker = self._get_or_create_tracker(
            session_id,
            session['name'],
            session.get('total_questions', 0)
        )

        # Load phase status from database into tracker
        for pn, phase_data in session['phases'].items():
            tracker.phases[pn].status = PhaseStatus(phase_data['status'])

        # Print pipeline before retry
        tracker.print_pipeline()

        # Execute retry
        result = self._retry_failed_questions_phase(session_id, session, tracker, config)

        # Print pipeline status after retry
        tracker.print_pipeline()

        return result


# Singleton instance
_session_service_instance = None


def get_session_service() -> SessionService:
    """Get or create the singleton session service instance."""
    global _session_service_instance
    if _session_service_instance is None:
        _session_service_instance = SessionService()
    return _session_service_instance
