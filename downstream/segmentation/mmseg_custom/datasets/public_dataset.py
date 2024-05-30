from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

@DATASETS.register_module()
class PublicDataset(BaseSegDataset): # Dataset train of PraNet
  METAINFO = dict(
        classes=('Background', 'Polyp'),
        palette=[[0, 0, 0], [111, 78, 55]])
  def __init__(self, **kwargs):
    super(PublicDataset, self).__init__(
      img_suffix='.png',
      seg_map_suffix='.png',
      **kwargs)
