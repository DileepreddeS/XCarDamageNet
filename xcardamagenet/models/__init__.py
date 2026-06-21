"""Model components for XCarDamageNet. Lazy-imported to allow independent module testing."""

from .backbone import DINOv2Backbone

try:
    from .physics_encoder import PhysicsTokenEncoder
except ImportError:
    pass

try:
    from .adaptive_attention import AdaptiveInspectionAttention
except ImportError:
    pass

try:
    from .contrastive import ContrastiveDamageModule
except ImportError:
    pass

try:
    from .neck import DamageAwareNeck
except ImportError:
    pass

try:
    from .head import ConfidenceGatedMultiTaskHead
except ImportError:
    pass

try:
    from .xcardamagenet import XCarDamageNet
except ImportError:
    pass
