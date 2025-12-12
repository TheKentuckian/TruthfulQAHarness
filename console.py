#!/usr/bin/env python
"""
Simple interactive console application for TruthfulQA Harness.

This console app provides a simplified interface to:
- Run evaluation sessions through a 4-phase workflow
- View session results
- List and manage sessions
"""
import os
import sys
from typing import Optional, Dict, Any

# Import backend modules
from backend.config import settings
from backend.services.dataset_loader import TruthfulQALoader
from backend.services.session_service import get_session_service
from backend.services.database import get_database


class ConsoleApp:
    """Main console application class."""

    def __init__(self):
        """Initialize the console app."""
        self.dataset_loader = TruthfulQALoader()
        self.session_service = get_session_service()
        self.db = get_database()

    def print_header(self, text: str):
        """Print a formatted header."""
        print(f"\n{'='*60}")
        print(f"  {text}")
        print(f"{'='*60}\n")

    def print_menu(self, title: str, options: list):
        """Print a menu with options."""
        print(f"\n{title}")
        print("-" * 40)
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")
        print("0. Back/Exit")
        print()

    def get_choice(self, max_choice: int) -> int:
        """Get user choice from menu."""
        while True:
            try:
                choice = int(input("Enter your choice: "))
                if 0 <= choice <= max_choice:
                    return choice
                print(f"Please enter a number between 0 and {max_choice}")
            except ValueError:
                print("Please enter a valid number")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                return 0

    def create_new_session(self):
        """Create a new evaluation session."""
        self.print_header("Create New Session")

        try:
            name = input("Enter session name: ").strip()
            if not name:
                print("Session name cannot be empty")
                return

            notes = input("Enter optional notes (press Enter to skip): ").strip() or None

            session = self.session_service.create_session(name=name, notes=notes)
            print(f"\nSession created successfully!")
            print(f"Session ID: {session['id']}")
            print(f"Name: {session['name']}")

        except Exception as e:
            print(f"Error creating session: {e}")

    def list_sessions(self, select_mode: bool = False) -> Optional[int]:
        """List all sessions and optionally allow selection."""
        self.print_header("Sessions")

        try:
            result = self.session_service.list_sessions(limit=100, offset=0)
            sessions = result

            if not sessions:
                print("No sessions found. Create a new session first.")
                return None

            print(f"{'ID':<6} {'Name':<30} {'Status':<15} {'Created':<20}")
            print("-" * 80)

            for session in sessions:
                session_id = session.get('id', 'N/A')
                name = session.get('name', 'N/A')[:28]
                status = session.get('status', 'N/A')
                created = session.get('created_at', 'N/A')[:19]
                print(f"{session_id:<6} {name:<30} {status:<15} {created:<20}")

            if select_mode:
                print("\n")
                try:
                    session_id = int(input("Enter session ID to select (0 to cancel): "))
                    if session_id == 0:
                        return None
                    # Verify session exists
                    if any(s.get('id') == session_id for s in sessions):
                        return session_id
                    else:
                        print(f"Session {session_id} not found")
                        return None
                except ValueError:
                    print("Invalid session ID")
                    return None

        except Exception as e:
            print(f"Error listing sessions: {e}")
            return None

    def run_session_workflow(self):
        """Run a complete session through all phases."""
        self.print_header("Run Session Workflow")

        # Select or create session
        print("1. Create new session")
        print("2. Use existing session")
        choice = self.get_choice(2)

        if choice == 0:
            return
        elif choice == 1:
            self.create_new_session()
            session_id = self.list_sessions(select_mode=True)
        else:
            session_id = self.list_sessions(select_mode=True)

        if not session_id:
            return

        # Phase 1: Gather Questions
        print("\n--- Phase 1: Gather Questions ---")
        sample_size = int(input("Number of questions to sample (default 10): ") or "10")
        use_seed = input("Use random seed for reproducibility? (y/n): ").lower() == 'y'
        seed = int(input("Enter seed: ")) if use_seed else None

        gather_config = {
            'sample_size': sample_size,
            'seed': seed,
            'use_all': False
        }

        print("\nRunning Phase 1...")
        try:
            self.session_service.run_phase(session_id, 1, gather_config)
            print("Phase 1 completed successfully")
        except Exception as e:
            print(f"Error in Phase 1: {e}")
            return

        # Phase 2: Generate Responses
        print("\n--- Phase 2: Generate Responses ---")
        print("LLM Provider:")
        print("1. Claude (requires ANTHROPIC_API_KEY)")
        print("2. LM Studio (local)")
        provider_choice = self.get_choice(2)

        if provider_choice == 0:
            return

        provider = "claude" if provider_choice == 1 else "lm_studio"

        generate_config = {
            'provider': provider,
            'max_tokens': int(input("Max tokens (default 1024): ") or "1024"),
            'temperature': float(input("Temperature (default 1.0): ") or "1.0"),
        }

        if provider == "lm_studio":
            generate_config['lm_studio_url'] = input("LM Studio URL (default http://localhost:1234/v1): ") or "http://localhost:1234/v1"
            generate_config['qwen_thinking'] = input("Enable Qwen thinking mode? (y/n): ").lower() == 'y'

        print("\nRunning Phase 2...")
        try:
            self.session_service.run_phase(session_id, 2, generate_config)
            print("Phase 2 completed successfully")
        except Exception as e:
            print(f"Error in Phase 2: {e}")
            return

        # Phase 3: Self-Correction (Optional)
        print("\n--- Phase 3: Self-Correction (Optional) ---")
        print("Correction methods:")
        print("1. None (skip correction)")
        print("2. Chain of Thought")
        print("3. Critique-based")
        print("4. Reward/Feedback")
        correction_choice = self.get_choice(4)

        if correction_choice == 0:
            return

        methods = ['none', 'chain_of_thought', 'critique', 'reward_feedback']
        method = methods[correction_choice - 1]

        correct_config = {
            'method': method,
            'provider': provider,
            'max_tokens': generate_config['max_tokens'],
            'temperature': generate_config['temperature'],
            'skip_threshold': 0.9
        }

        if provider == "lm_studio":
            correct_config['lm_studio_url'] = generate_config.get('lm_studio_url', 'http://localhost:1234/v1')

        if method != 'none':
            print("\nRunning Phase 3...")
            try:
                self.session_service.run_phase(session_id, 3, correct_config)
                print("Phase 3 completed successfully")
            except Exception as e:
                print(f"Error in Phase 3: {e}")
                return

        # Phase 4: Validation
        print("\n--- Phase 4: Validation ---")
        print("Verifier:")
        print("1. Simple Text (word overlap)")
        print("2. Word Similarity (TF-IDF)")
        print("3. LLM Judge")
        verifier_choice = self.get_choice(3)

        if verifier_choice == 0:
            return

        verifiers = ['simple_text', 'word_similarity', 'llm_judge']
        verifier_type = verifiers[verifier_choice - 1]

        validate_config = {
            'verifier_type': verifier_type,
        }

        if verifier_type == 'llm_judge':
            print("Judge LLM:")
            print("1. Claude")
            print("2. LM Studio (local)")
            judge_choice = self.get_choice(2)
            if judge_choice == 0:
                return

            validate_config['judge_provider'] = "claude" if judge_choice == 1 else "lm_studio"
            if validate_config['judge_provider'] == "lm_studio":
                validate_config['judge_url'] = input("Judge LM Studio URL (default http://localhost:1234/v1): ") or "http://localhost:1234/v1"

        print("\nRunning Phase 4...")
        try:
            self.session_service.run_phase(session_id, 4, validate_config)
            print("Phase 4 completed successfully")
        except Exception as e:
            print(f"Error in Phase 4: {e}")
            return

        # Show results
        print("\n" + "="*60)
        print("SESSION COMPLETED")
        print("="*60)
        self.view_session_results(session_id)

    def view_session_results(self, session_id: Optional[int] = None):
        """View results for a specific session."""
        if not session_id:
            self.print_header("View Session Results")
            session_id = self.list_sessions(select_mode=True)

        if not session_id:
            return

        try:
            session = self.session_service.get_session(session_id)

            print(f"\nSession: {session['name']}")
            print(f"Status: {session.get('status', 'N/A')}")
            print(f"Total Questions: {session.get('total_questions', 0)}")
            print(f"\nPhases:")

            phases = session.get('phases', {})
            for phase_num in sorted(phases.keys(), key=int):
                phase_data = phases[phase_num]
                print(f"\nPhase {phase_num}: {phase_data.get('status', 'N/A')}")

                results = phase_data.get('results', {})
                if results:
                    for key, value in results.items():
                        print(f"  {key}: {value}")

            # Get responses for validation phase
            if '4' in phases and phases['4'].get('status') == 'completed':
                responses = self.session_service.get_session_responses(session_id, phase_number=4)
                if responses:
                    print(f"\nValidation Results:")
                    truthful_count = sum(1 for r in responses if r.get('is_truthful'))
                    print(f"  Truthful answers: {truthful_count}/{len(responses)}")
                    print(f"  Accuracy: {truthful_count/len(responses)*100:.1f}%")

        except Exception as e:
            print(f"Error viewing session results: {e}")

    def delete_session(self):
        """Delete a session."""
        self.print_header("Delete Session")

        session_id = self.list_sessions(select_mode=True)
        if not session_id:
            return

        confirm = input(f"\nAre you sure you want to delete session {session_id}? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Deletion cancelled")
            return

        try:
            self.session_service.delete_session(session_id)
            print(f"Session {session_id} deleted successfully")
        except Exception as e:
            print(f"Error deleting session: {e}")

    def dataset_info(self):
        """Display dataset information."""
        self.print_header("Dataset Information")

        try:
            info = self.dataset_loader.get_dataset_info()
            print(f"Dataset: {info['dataset_name']}")
            print(f"Split: {info['split']}")
            print(f"Total Questions: {info['total_questions']}")
        except Exception as e:
            print(f"Error getting dataset info: {e}")

    def main_menu(self):
        """Display main menu and handle navigation."""
        while True:
            self.print_header("TruthfulQA Evaluation Harness")

            options = [
                "Run Session Workflow (Phases 1-4)",
                "View Session Results",
                "List All Sessions",
                "Delete Session",
                "Dataset Information",
            ]

            self.print_menu("Main Menu", options)

            choice = self.get_choice(len(options))

            if choice == 0:
                print("\nGoodbye!")
                break
            elif choice == 1:
                self.run_session_workflow()
            elif choice == 2:
                self.view_session_results()
            elif choice == 3:
                self.list_sessions()
                input("\nPress Enter to continue...")
            elif choice == 4:
                self.delete_session()
            elif choice == 5:
                self.dataset_info()
                input("\nPress Enter to continue...")

    def run(self):
        """Run the console application."""
        try:
            self.main_menu()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            sys.exit(0)


def main():
    """Main entry point."""
    # Check for API key if needed
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("Warning: ANTHROPIC_API_KEY not set in environment")
        print("You can still use LM Studio as the LLM provider\n")

    app = ConsoleApp()
    app.run()


if __name__ == "__main__":
    main()
