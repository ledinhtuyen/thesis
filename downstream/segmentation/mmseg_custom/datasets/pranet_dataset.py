from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

@DATASETS.register_module()
class PraNet_Dataset(BaseSegDataset):
  METAINFO = dict(
        classes=('Polyp',),
        palette=[[255, 255, 255],])
  def __init__(self, **kwargs):
    super(PraNet_Dataset, self).__init__(
      img_suffix='.png',
      seg_map_suffix='.png',
      **kwargs)
