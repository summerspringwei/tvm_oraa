"""We extract one feature vector for each GPU block,
so we call this feature as "per-block" feature.
"""
from tvm._ffi import register_object

from .. import _ffi_api
from .feature_extractor import FeatureExtractor


@register_object("meta_schedule.PerBlockFeature")
class PerBlockFeature(FeatureExtractor):
    """PerBlockFeature extracts one feature vector per BufferStoreNode

    Parameters
    ----------
    extract_workload : bool
        Whether to extract features in the workload in tuning context or not.
    """

    extract_workload: bool
    """Whether to extract features in the workload in tuning context or not."""
    feature_vector_length: int
    """Length of the feature vector."""

    def __init__(
        self,
        extract_workload: bool = False,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.FeatureExtractorPerBlockFeature,  # type: ignore # pylint: disable=no-member
            extract_workload,
        )
