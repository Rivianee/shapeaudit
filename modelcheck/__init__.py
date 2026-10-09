try:
    from importlib.metadata import version as _v
    try:
        __version__ = _v("shapeaudit")
    except Exception:                      # instalado pelo nome anterior
        __version__ = _v("modelguard-ml")
except Exception:
    __version__ = "0.0.0+dev"


from .ale_detector import ALEShapeDetector, ale_shape_profile

from .audit import audit
from .geometry import GeometryDivergenceDetector

from .paired import PairedDivergenceAnalyzer

from .rhale import RHALEDetector, rhale_profile
from .zest import ArchitectureAgnosticComparator

from .trending import ShapeTrendingDetector
from .ood import OODBehavioralDetector

from .canary import GeometricCanaryDetector
from .explorer import InteractiveDivergenceExplorer

from .topology import TopologicalDriftDetector
from .stability import SupervisedStabilityAnalyzer
from .gtsf import GTSF

from .orchestrator import ModelGuardOrchestrator

from .importance import ImportanceCollapseDetector

from .confidence import score_with_ci, compare_scores_ci

from .baselines import run_all_baselines

from .shapedist import shape_cost, severity, profile_distance

from .ale_fast import ale_curve_fast, model_embedding_ale, feature_kind

from .universal import UniversalCanaryDetector, TopologicalDriftGudhi


from .ale_selective import SelectiveALEDetector

from .validation import TrainServingValidator

from .guard import ModelGuard
from .api import Guard, AuditResult

from .leakage_temporal import TemporalLeakageDetector

from .psi_shap import PSIShapDetector

from .backends import importance as feature_importance, BACKENDS

from modelcheck import snapshot  # noqa: E402,F401
