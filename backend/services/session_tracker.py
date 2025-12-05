"""Session tracking and estimation for testing sessions."""
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class PhaseType(str, Enum):
    """Types of session phases."""
    GATHER = "gather"
    GENERATE = "generate"
    CORRECT = "correct"
    VALIDATE = "validate"


class PhaseStatus(str, Enum):
    """Status of a phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class CorrectionMethod(str, Enum):
    """Available self-correction methods."""
    NONE = "none"
    CRITIQUE = "critique"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    REWARD_FEEDBACK = "reward_feedback"


@dataclass
class TimeEstimator:
    """Estimates remaining time based on rolling average with warmup."""

    warmup_items: int = 3
    item_times: List[float] = field(default_factory=list)

    def add_sample(self, seconds: float):
        """Add a timing sample."""
        self.item_times.append(seconds)

    def get_average(self) -> float:
        """Get current average time per item."""
        if not self.item_times:
            return 0.0
        return sum(self.item_times) / len(self.item_times)

    def estimate_remaining(self, remaining_items: int) -> float:
        """Estimate remaining time in seconds."""
        if not self.item_times:
            return 0.0

        if len(self.item_times) < self.warmup_items:
            # Not enough data - use simple average
            avg = self.get_average()
            return avg * remaining_items

        # Use weighted recent average (last 5 items weighted more)
        recent = self.item_times[-5:] if len(self.item_times) >= 5 else self.item_times
        weighted_avg = sum(recent) / len(recent)

        # Add 10% buffer for variance
        return weighted_avg * remaining_items * 1.1

    @staticmethod
    def format_time(seconds: float) -> str:
        """Format seconds as human-readable string."""
        if seconds < 0:
            return "0s"
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"


@dataclass
class PhaseProgress:
    """Tracks progress of a single phase."""

    phase_number: int
    phase_type: PhaseType
    status: PhaseStatus = PhaseStatus.PENDING

    # Progress counters
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimator: TimeEstimator = field(default_factory=TimeEstimator)

    # Configuration used for this phase
    config: Dict[str, Any] = field(default_factory=dict)

    # Results summary
    results_summary: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def percent_complete(self) -> float:
        """Get completion percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.completed_items / self.total_items) * 100

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        if not self.started_at:
            return 0.0
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()

    @property
    def estimated_remaining_seconds(self) -> float:
        """Get estimated remaining time."""
        remaining = self.total_items - self.completed_items
        return self.estimator.estimate_remaining(remaining)

    @property
    def estimated_total_seconds(self) -> float:
        """Get estimated total time."""
        return self.elapsed_seconds + self.estimated_remaining_seconds

    def start(self, total_items: int, config: Dict[str, Any] = None):
        """Start the phase."""
        self.status = PhaseStatus.RUNNING
        self.started_at = datetime.now()
        self.total_items = total_items
        self.completed_items = 0
        self.failed_items = 0
        self.skipped_items = 0
        self.config = config or {}
        self.estimator = TimeEstimator()

    def record_item(self, duration: float, success: bool = True, skipped: bool = False):
        """Record completion of an item."""
        if skipped:
            self.skipped_items += 1
        elif success:
            self.completed_items += 1
        else:
            self.failed_items += 1

        # Always increment completed for progress tracking
        if not skipped or success:
            self.estimator.add_sample(duration)

    def complete(self, results_summary: Dict[str, Any] = None):
        """Mark phase as complete."""
        self.status = PhaseStatus.COMPLETED
        self.completed_at = datetime.now()
        self.results_summary = results_summary or {}

    def fail(self, error: str):
        """Mark phase as failed."""
        self.status = PhaseStatus.FAILED
        self.completed_at = datetime.now()
        self.error = error

    def skip(self):
        """Mark phase as skipped."""
        self.status = PhaseStatus.SKIPPED
        self.completed_at = datetime.now()


@dataclass
class SessionTracker:
    """Tracks overall session state and progress."""

    session_id: int
    session_name: str
    total_questions: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    # Phase tracking (1-indexed for user clarity)
    phases: Dict[int, PhaseProgress] = field(default_factory=dict)

    # Console logger reference
    logger: Optional['SessionConsoleLogger'] = None

    def __post_init__(self):
        """Initialize phases."""
        if not self.phases:
            self.phases = {
                1: PhaseProgress(1, PhaseType.GATHER),
                2: PhaseProgress(2, PhaseType.GENERATE),
                3: PhaseProgress(3, PhaseType.CORRECT),
                4: PhaseProgress(4, PhaseType.VALIDATE),
            }
        if self.logger is None:
            self.logger = SessionConsoleLogger()

    def get_phase(self, phase_num: int) -> PhaseProgress:
        """Get a phase by number."""
        return self.phases.get(phase_num)

    def start_phase(self, phase_num: int, total_items: int, config: Dict[str, Any] = None):
        """Start a phase."""
        phase = self.phases[phase_num]
        phase.start(total_items, config)
        self.logger.print_phase_start(phase, config or {})

    def update_progress(self, phase_num: int, item_idx: int,
                       question: str, result: Dict[str, Any],
                       duration: float, success: bool = True, skipped: bool = False):
        """Update progress for a phase."""
        phase = self.phases[phase_num]
        phase.record_item(duration, success, skipped)
        self.logger.print_item_complete(phase, item_idx, question, result, duration, skipped)
        self.logger.print_progress_update(phase)

    def complete_phase(self, phase_num: int, summary: Dict[str, Any] = None):
        """Complete a phase."""
        phase = self.phases[phase_num]
        phase.complete(summary)
        self.logger.print_phase_summary(phase)

    def fail_phase(self, phase_num: int, error: str):
        """Mark a phase as failed."""
        phase = self.phases[phase_num]
        phase.fail(error)
        self.logger.print_phase_error(phase, error)

    def skip_phase(self, phase_num: int):
        """Skip a phase."""
        phase = self.phases[phase_num]
        phase.skip()
        self.logger.print_phase_skipped(phase)

    def print_header(self):
        """Print session header."""
        self.logger.print_session_header(self)

    def print_pipeline(self):
        """Print phase pipeline status."""
        self.logger.print_phase_pipeline(self)

    def print_summary(self):
        """Print session summary."""
        self.logger.print_session_summary(self)

    def get_current_phase(self) -> Optional[PhaseProgress]:
        """Get the currently running phase."""
        for phase in self.phases.values():
            if phase.status == PhaseStatus.RUNNING:
                return phase
        return None

    def get_next_pending_phase(self) -> Optional[PhaseProgress]:
        """Get the next pending phase."""
        for i in range(1, 5):
            if self.phases[i].status == PhaseStatus.PENDING:
                return self.phases[i]
        return None

    def is_complete(self) -> bool:
        """Check if all phases are complete."""
        for phase in self.phases.values():
            if phase.status not in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED):
                return False
        return True


class SessionConsoleLogger:
    """Handles all console output for session tracking."""

    # ANSI color codes
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'dim': '\033[2m',
        'green': '\033[92m',
        'red': '\033[91m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'cyan': '\033[96m',
        'white': '\033[97m',
    }

    # Status symbols
    SYMBOLS = {
        'pending': '○',
        'running': '▶',
        'completed': '✓',
        'failed': '✗',
        'skipped': '—',
    }

    def __init__(self, use_colors: bool = True):
        """Initialize the logger."""
        self.use_colors = use_colors and sys.stdout.isatty()
        self._last_progress_line = False

    def _color(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if self.use_colors and color in self.COLORS:
            return f"{self.COLORS[color]}{text}{self.COLORS['reset']}"
        return text

    def _symbol(self, status: str) -> str:
        """Get symbol for status with color."""
        symbol = self.SYMBOLS.get(status, '?')
        color_map = {
            'pending': 'dim',
            'running': 'blue',
            'completed': 'green',
            'failed': 'red',
            'skipped': 'yellow',
        }
        return self._color(symbol, color_map.get(status, 'white'))

    def _clear_progress_line(self):
        """Clear the last progress line if needed."""
        if self._last_progress_line and self.use_colors:
            # Move up one line and clear
            sys.stdout.write('\033[1A\033[2K')
            self._last_progress_line = False

    def print_session_header(self, session: 'SessionTracker'):
        """Print session banner at start."""
        width = 80
        print()
        print("═" * width)
        title = f" SESSION #{session.session_id}: \"{session.session_name}\""
        print(self._color(title, 'bold'))
        print(f" Created: {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}", end="")
        if session.total_questions > 0:
            print(f" | Questions: {session.total_questions}")
        else:
            print()
        print("═" * width)
        print()

    def print_phase_pipeline(self, session: 'SessionTracker'):
        """Print the 4-phase status bar."""
        print(" PHASE PIPELINE STATUS")

        # Build phase boxes
        phases = [session.phases[i] for i in range(1, 5)]
        names = ["GATHER", "GENERATE", "CORRECT", "VALIDATE"]

        # Top border
        print(" ┌" + "┬".join(["──────────"] * 4) + "┐")

        # Phase names
        name_row = " │"
        for name in names:
            name_row += f" {name:^8} │"
        print(name_row)

        # Status symbols
        status_row = " │"
        for phase in phases:
            symbol = self._symbol(phase.status.value)
            status_row += f"    {symbol}     │"
        print(status_row)

        # Timing/status info
        info_row = " │"
        for phase in phases:
            if phase.status == PhaseStatus.COMPLETED:
                info = TimeEstimator.format_time(phase.elapsed_seconds)
            elif phase.status == PhaseStatus.RUNNING:
                info = "..."
            elif phase.status == PhaseStatus.FAILED:
                info = "failed"
            elif phase.status == PhaseStatus.SKIPPED:
                info = "skipped"
            else:
                info = "pending"
            info_row += f" {info:^8} │"
        print(info_row)

        # Bottom border
        print(" └" + "┴".join(["──────────"] * 4) + "┘")
        print()

    def print_phase_start(self, phase: PhaseProgress, config: Dict[str, Any]):
        """Print phase header when starting."""
        self._clear_progress_line()

        width = 80
        print("─" * width)

        phase_names = {
            PhaseType.GATHER: "GATHER",
            PhaseType.GENERATE: "GENERATE",
            PhaseType.CORRECT: "CORRECT",
            PhaseType.VALIDATE: "VALIDATE",
        }

        title = f" PHASE {phase.phase_number}: {phase_names[phase.phase_type]}"

        # Add config details
        if phase.phase_type == PhaseType.GENERATE:
            provider = config.get('provider', 'unknown')
            model = config.get('model', 'unknown')
            title += f" ({provider}: {model})"
        elif phase.phase_type == PhaseType.CORRECT:
            method = config.get('method', 'unknown')
            title += f" ({method})"
        elif phase.phase_type == PhaseType.VALIDATE:
            provider = config.get('provider', 'LM Studio')
            model = config.get('model', 'unknown')
            title += f" ({provider}: {model})"

        print(self._color(title, 'bold'))
        print("─" * width)

    def print_progress_update(self, phase: PhaseProgress):
        """Print/update progress bar."""
        self._clear_progress_line()

        # Build progress bar
        bar_width = 30
        filled = int((phase.completed_items / max(phase.total_items, 1)) * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)

        percent = phase.percent_complete
        progress = f" Progress: [{bar}] {phase.completed_items}/{phase.total_items} ({percent:.0f}%)"

        # Add timing info
        elapsed = TimeEstimator.format_time(phase.elapsed_seconds)
        remaining = TimeEstimator.format_time(phase.estimated_remaining_seconds)
        timing = f" | Elapsed: {elapsed} | Est. remaining: ~{remaining}"

        print(progress + timing)
        self._last_progress_line = True

    def print_item_complete(self, phase: PhaseProgress, item_idx: int,
                           question: str, result: Dict[str, Any],
                           duration: float, skipped: bool = False):
        """Print single item completion line."""
        self._clear_progress_line()

        timestamp = datetime.now().strftime("%H:%M:%S")
        duration_str = f"{duration:.1f}s"

        # Truncate question for display
        max_q_len = 50
        q_display = question[:max_q_len] + "..." if len(question) > max_q_len else question

        if skipped:
            symbol = self._color("○", "yellow")
            status_info = self._color("Skipped", "yellow")
            reason = result.get('skip_reason', 'already sufficient')
            print(f" [{timestamp}] {symbol} Q{item_idx:<3} ({duration_str}) {status_info} - {reason}")
        else:
            # Determine success/failure based on phase type
            if phase.phase_type == PhaseType.VALIDATE:
                is_truthful = result.get('is_truthful', False)
                confidence = result.get('confidence', 0)
                if is_truthful:
                    symbol = self._color("✓", "green")
                    status = self._color("TRUTHFUL", "green")
                else:
                    symbol = self._color("✗", "red")
                    status = self._color("UNTRUTHFUL", "red")
                print(f" [{timestamp}] {symbol} Q{item_idx:<3} ({duration_str}) {status} (confidence: {confidence:.2f})")

            elif phase.phase_type == PhaseType.CORRECT:
                before_conf = result.get('before_confidence', 0)
                after_conf = result.get('after_confidence', 0)
                improvement = after_conf - before_conf
                symbol = self._color("✓", "green")
                if improvement > 0:
                    imp_str = self._color(f"+{improvement:.2f}", "green")
                else:
                    imp_str = f"{improvement:.2f}"
                print(f" [{timestamp}] {symbol} Q{item_idx:<3} ({duration_str}) confidence: {before_conf:.2f} → {after_conf:.2f} ({imp_str})")

            elif phase.phase_type == PhaseType.GENERATE:
                symbol = self._color("✓", "green")
                print(f" [{timestamp}] {symbol} Q{item_idx:<3} ({duration_str}) \"{q_display}\"")

            else:
                symbol = self._color("✓", "green")
                print(f" [{timestamp}] {symbol} Q{item_idx:<3} ({duration_str})")

    def print_phase_summary(self, phase: PhaseProgress):
        """Print phase completion summary."""
        self._clear_progress_line()

        print()
        elapsed = TimeEstimator.format_time(phase.elapsed_seconds)
        status_symbol = self._color("✓", "green")

        phase_names = {
            PhaseType.GATHER: "Gather",
            PhaseType.GENERATE: "Generate",
            PhaseType.CORRECT: "Correct",
            PhaseType.VALIDATE: "Validate",
        }

        print(f" {status_symbol} Phase {phase.phase_number} ({phase_names[phase.phase_type]}) Complete ({elapsed})")

        # Print phase-specific summary
        summary = phase.results_summary

        if phase.phase_type == PhaseType.GATHER:
            categories = summary.get('categories', {})
            if categories:
                cat_str = ", ".join([f"{k} ({v})" for k, v in list(categories.items())[:5]])
                if len(categories) > 5:
                    cat_str += f", +{len(categories) - 5} more"
                print(f"   Categories: {cat_str}")

        elif phase.phase_type == PhaseType.GENERATE:
            total = summary.get('total_responses', 0)
            avg_time = summary.get('avg_response_time', 0)
            print(f"   Total responses: {total}")
            print(f"   Avg response time: {avg_time:.2f}s")

        elif phase.phase_type == PhaseType.CORRECT:
            applied = summary.get('corrections_applied', 0)
            total = summary.get('total', 0)
            skipped = summary.get('skipped', 0)
            avg_improvement = summary.get('avg_improvement', 0)
            print(f"   Corrections applied: {applied}/{total} ({100*applied/max(total,1):.0f}%)")
            print(f"   Skipped (high confidence): {skipped}")
            if applied > 0:
                print(f"   Avg improvement: +{avg_improvement:.2f} confidence")

        elif phase.phase_type == PhaseType.VALIDATE:
            truthful = summary.get('truthful_count', 0)
            total = summary.get('total', 0)
            accuracy = summary.get('accuracy', 0)
            avg_conf = summary.get('avg_confidence', 0)
            print(f"   Truthful: {truthful}/{total} ({accuracy:.1f}%)")
            print(f"   Avg confidence: {avg_conf:.2f}")

        print()

    def print_phase_error(self, phase: PhaseProgress, error: str):
        """Print phase error."""
        self._clear_progress_line()

        print()
        symbol = self._color("✗", "red")
        print(f" {symbol} Phase {phase.phase_number} Failed")
        print(f"   Error: {error}")
        print()

    def print_phase_skipped(self, phase: PhaseProgress):
        """Print phase skipped message."""
        self._clear_progress_line()

        print()
        symbol = self._color("—", "yellow")
        phase_names = {
            PhaseType.GATHER: "Gather",
            PhaseType.GENERATE: "Generate",
            PhaseType.CORRECT: "Correct",
            PhaseType.VALIDATE: "Validate",
        }
        print(f" {symbol} Phase {phase.phase_number} ({phase_names[phase.phase_type]}) Skipped")
        print()

    def print_session_summary(self, session: 'SessionTracker'):
        """Print final session summary with all phases."""
        self._clear_progress_line()

        width = 80
        print()
        print("═" * width)
        print(self._color(" SESSION COMPLETE", "bold"))
        print("═" * width)
        print()

        # Print phase pipeline one more time
        self.print_phase_pipeline(session)

        # Calculate total duration
        total_duration = 0
        for phase in session.phases.values():
            if phase.status == PhaseStatus.COMPLETED:
                total_duration += phase.elapsed_seconds

        print(f" Total Duration: {TimeEstimator.format_time(total_duration)}")
        print()

        # Print final results if validation was completed
        validate_phase = session.phases[4]
        if validate_phase.status == PhaseStatus.COMPLETED:
            summary = validate_phase.results_summary
            print(" FINAL RESULTS")
            print(" ┌" + "─" * 47 + "┐")

            accuracy = summary.get('accuracy', 0)
            truthful = summary.get('truthful_count', 0)
            total = summary.get('total', 0)
            avg_conf = summary.get('avg_confidence', 0)

            print(f" │  Accuracy:        {accuracy:5.1f}% ({truthful}/{total} truthful){' ' * 10}│")
            print(f" │  Avg Confidence:  {avg_conf:.2f}{' ' * 27}│")

            # Include correction info if phase 3 was run
            correct_phase = session.phases[3]
            if correct_phase.status == PhaseStatus.COMPLETED:
                corr_summary = correct_phase.results_summary
                corr_rate = corr_summary.get('corrections_applied', 0) / max(corr_summary.get('total', 1), 1) * 100
                avg_imp = corr_summary.get('avg_improvement', 0)
                print(f" │  Correction Rate: {corr_rate:5.1f}%{' ' * 26}│")
                print(f" │  Avg Improvement: +{avg_imp:.2f} confidence{' ' * 16}│")

            print(" └" + "─" * 47 + "┘")

        print()
        print(f" Session saved to database (ID: {session.session_id})")
        print("═" * width)
        print()
